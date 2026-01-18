import contextlib
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily, is_fake
from torch._dispatch.python import enable_python_dispatcher, enable_pre_dispatch
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass
import weakref
import operator
from torch.utils._stats import count
import logging
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
from .sym_node import SymNode
from ._sym_dispatch_mode import SymDispatchMode
from torch.fx import Proxy
import torch.fx.traceback as fx_traceback
from torch import SymInt, SymFloat, SymBool
from torch.utils.weak import WeakTensorKeyDictionary
def proxy_call(proxy_mode, func, pre_dispatch, args, kwargs):
    unrecognized_types = []

    def can_handle_tensor(x):
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) in (torch._subclasses.FakeTensor,)
        if not r:
            unrecognized_types.append(type(x))
        return r
    if not pytree.tree_all_only(torch.Tensor, can_handle_tensor, (args, kwargs)):
        not_implemented_log.debug('ProxyTensorMode tensors without proxy had unrecognized subclasses: %s', unrecognized_types)
        return NotImplemented
    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        return r
    if not pre_dispatch and func not in [torch.ops.aten.size.default, torch.ops.aten.stride.default, torch.ops.aten.storage_offset.default]:
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r
    tracer = proxy_mode.tracer
    f_args, f_kwargs = pytree.tree_map_only(torch.Tensor, fetch_tensor_proxy(tracer), (args, kwargs))
    all_constant = pytree.tree_all_only(_ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs)) and pytree.tree_all_only((SymInt, SymFloat, SymBool), lambda _: False, (args, kwargs))
    if torch.Tag.data_dependent_output in func.tags:
        if all_constant:
            const_args, const_kwargs = pytree.tree_map_only(_ProxyTensor, lambda t: t.constant, (f_args, f_kwargs))
            with maybe_disable_fake_tensor_mode():
                return func(*const_args, **const_kwargs)
        if proxy_mode._error_on_data_dependent_ops and pytree.tree_all_only(torch.Tensor, lambda t: not is_fake(t), (args, kwargs)):
            raise RuntimeError(f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! It's likely that this is caused by data-dependent control flow or similar.  It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' in your make_fx call.")
    proxy_args, proxy_kwargs = pytree.tree_map_only((SymInt, SymFloat, SymBool), fetch_sym_proxy(proxy_mode.tracer), pytree.tree_map_only(_ProxyTensor, lambda e: e.proxy, (f_args, f_kwargs)))
    if func is torch.ops.aten.lift_fresh.default:
        func = torch.ops.aten.lift_fresh_copy.default
    proxy_out = proxy_mode.tracer.create_proxy('call_function', func, proxy_args, proxy_kwargs, name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__))
    if func.overloadpacket.__name__[-1] == '_' and func.overloadpacket.__name__[0] != '_':
        if isinstance(args[0], List):
            for i, a in enumerate(args[0]):
                a.proxy = proxy_out[0][i]
        else:
            args[0].proxy = proxy_out
    out = func(*args, **kwargs)
    any_constant = pytree.tree_any_only(_ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs))
    constant = None
    if func is torch.ops.aten.lift_fresh_copy.default and out.numel() <= CONSTANT_NUMEL_LIMIT:
        with maybe_disable_fake_tensor_mode():
            constant = args[0].clone()
    elif torch.Tag.nondeterministic_seeded not in func.tags and all_constant and any_constant and pytree.tree_all_only(torch.Tensor, lambda t: t.numel() <= CONSTANT_NUMEL_LIMIT, out):
        with maybe_disable_fake_tensor_mode():
            const_args, const_kwargs = pytree.tree_map_only(_ProxyTensor, lambda t: t.constant, (f_args, f_kwargs))
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None
    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    return out