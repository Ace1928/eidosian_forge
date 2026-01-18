from __future__ import annotations
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import (
from typing_extensions import TypeGuard
import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type
def fx_to_pattern(gm, ignore_types=(), argnames=(), scalar_workaround=(), exclusive_arg_names=()) -> PatternExpr:
    """
    Convert an FX graph into a PatternExpr.  This is useful for simple
    patterns that can only match single functions and fixed-length lists.
    """
    scalar_workaround = scalar_workaround or {}
    inv_scalar_workaround = {v: k for k, v in scalar_workaround.items()}
    assert len(inv_scalar_workaround) == len(scalar_workaround)

    def process_arg(x):
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in ignore_types:
            return Ignored()
        if isinstance(x, list) and all((isinstance(y, Ignored) for y in x)) and x:
            return Ignored()
        return x
    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        call_method = _not_implemented
        call_module = _not_implemented
        get_attr = _not_implemented

        def placeholder(self, target, args, kwargs):
            n = next(argnum)
            if n < len(argnames):
                name = argnames[n]
            elif argnames:
                assert target.startswith('tangent')
                name = target
            else:
                target = re.sub('_\\d+$', '', target)
                name = target
            if name in exclusive_arg_names:
                return ExclusiveKeywordArg(name)
            else:
                return KeywordArg(name)

        def call_function(self, target, args, kwargs):
            args, kwargs = pytree.tree_map(process_arg, (args, kwargs))
            if list in ignore_types:
                args = [process_arg(a) for a in args]
                kwargs = {k: process_arg(a) for k, a in kwargs.items()}
            return CallFunction(target, *args, **kwargs)

        def run_node(self, n):
            rv = super().run_node(n)
            if n.op == 'output' and isinstance(rv, tuple):
                assert len(rv) == len(n.args[0])
                for r, arg in zip(rv, n.args[0]):
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv
    pattern = Converter(gm).run()
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern