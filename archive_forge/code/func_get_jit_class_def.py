import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401
import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
from torch._jit_internal import (  # noqa: F401
from torch._sources import (
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace
def get_jit_class_def(cls, self_name):
    methods = inspect.getmembers(cls, predicate=lambda m: (inspect.ismethod(m) or inspect.isfunction(m)) and (not is_static_fn(cls, m.__name__)) and (m.__name__ in cls.__dict__) and (not _is_drop_fn(m)))

    def is_classmethod(fn):
        return inspect.ismethod(fn) and getattr(fn, '__self__', None) == cls
    sourcelines, file_lineno, filename = get_source_lines_and_file(cls, torch._C.ErrorReport.call_stack())
    source = ''.join(sourcelines)
    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)
    class_ast = py_ast.body[0]
    assert isinstance(class_ast, ast.ClassDef)
    if dataclasses.is_dataclass(cls):
        overrides = {method.name for method in class_ast.body if isinstance(method, ast.FunctionDef) and method.name in DATACLASS_MAGIC_METHODS}
        for i, (name, _) in enumerate(methods):
            synthesizer_fn = DATACLASS_MAGIC_METHODS.get(name)
            if synthesizer_fn and name not in overrides:
                parsed_def = synthesizer_fn(cls)
                methods[i] = (name, parsed_def)
                func = getattr(cls, name)
                _jit_internal.loader.cache(func, parsed_def.source)
    method_defs = [get_jit_def(obj, name, self_name=self_name, is_classmethod=is_classmethod(obj)) for name, obj in methods]
    properties = get_class_properties(cls, self_name)
    leading_whitespace_len = len(source.split('\n', 1)[0]) - len(dedent_src.split('\n', 1)[0])
    ctx = make_source_context(source, filename, file_lineno, leading_whitespace_len, False)
    assigns = get_class_assigns(ctx, class_ast)
    return build_class_def(ctx, class_ast, method_defs, properties, self_name, assigns)