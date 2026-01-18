import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def emit_node(node: Node):
    maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
    if verbose:
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx.experimental.proxy_tensor import py_sym_types
        from torch.fx.passes.shape_prop import TensorMetadata
        meta_val = node.meta.get('val', node.meta.get('tensor_meta', None))
        if isinstance(meta_val, FakeTensor):
            maybe_type_annotation = f': "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}"'
        elif isinstance(meta_val, py_sym_types):
            maybe_type_annotation = f': "Sym({meta_val})"'
        elif isinstance(meta_val, TensorMetadata):
            maybe_type_annotation = f': "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}"'
    if node.op == 'placeholder':
        assert isinstance(node.target, str)
        maybe_default_arg = '' if not node.args else f' = {_get_repr(node.args[0])}'
        free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
        raw_name = node.target.replace('*', '')
        if raw_name != repr(node):
            body.append(f'{repr(node)} = {raw_name}\n')
        return
    elif node.op == 'call_method':
        assert isinstance(node.target, str)
        body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.target)}({_format_args(node.args[1:], node.kwargs)})')
        return
    elif node.op == 'call_function':
        assert callable(node.target)
        if getattr(node.target, '__module__', '') == '_operator' and node.target.__name__ in magic_methods:
            assert isinstance(node.args, tuple)
            body.append(f'{repr(node)}{maybe_type_annotation} = {magic_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))}')
            return
        if getattr(node.target, '__module__', '') == '_operator' and node.target.__name__ in inplace_methods:
            body.append(f'{inplace_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))};  {repr(node)}{maybe_type_annotation} = {_get_repr(node.args[0])}')
            return
        qualified_name = _get_qualified_name(node.target)
        global_name = add_global(qualified_name, node.target)
        if global_name == 'getattr' and isinstance(node.args, tuple) and isinstance(node.args[1], str) and node.args[1].isidentifier() and (len(node.args) == 2):
            body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.args[1])}')
            return
        body.append(f'{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})')
        if node.meta.get('is_wrapped', False):
            wrapped_fns.setdefault(global_name)
        return
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
        return
    elif node.op == 'get_attr':
        assert isinstance(node.target, str)
        body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}')
        return
    elif node.op == 'output':
        if node.type is not None:
            maybe_return_annotation[0] = f' -> {type_repr(node.type)}'
        body.append(self.generate_output(node.args[0]))
        return
    raise NotImplementedError(f'node: {node.op} {node.target}')