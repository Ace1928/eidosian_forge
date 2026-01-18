import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
def run_fallback_kernel(fake_mode, func, flat_args, args_spec, orig_not_implemented_exception):
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception
    inp_impls = {}
    with no_dispatch():

        def to_real_tensor(e):
            if fake_mode.is_our_fake(e):
                out = torch.zeros_like(e, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                inp_impls[id(out)] = e
                return out
            return e
        flat_args = [to_real_tensor(a) for a in flat_args]
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
        r = func(*args, **kwargs)
    tensor_impls = set()
    storages = set()
    for e in flat_args:
        if isinstance(e, torch.Tensor):
            if not e.is_sparse:
                storages.add(e._typed_storage()._cdata)

    def map_out(e):
        if id(e) not in inp_impls and (isinstance(e, torch.Tensor) and (not e.is_sparse) and (e._typed_storage()._cdata in storages)):
            raise orig_not_implemented_exception
        if isinstance(e, torch.Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter(fake_mode, e)
        else:
            return e
    return pytree.tree_map(map_out, r)