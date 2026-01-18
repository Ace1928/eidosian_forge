import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import types
import typing
import warnings
import weakref
from textwrap import dedent
from typing import (  # noqa: F401
import torch
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future
def get_type_hint_captures(fn):
    """
    Get a dictionary containing type resolution mappings necessary to resolve types
    for the literal annotations on 'fn'. These are not considered to be closed-over by fn
    and must be obtained separately (e.g. using this function).

    Args:
        fn: A callable.
    Returns:
        A Dict[str, Any] containing a mapping from the literal annotations used on
        fn to the Python objects they refer to.
    """
    src = loader.get_source(fn)
    if src is None:
        src = inspect.getsource(fn)
    signature = inspect.signature(fn)
    name_to_type = {name: parameter.annotation for name, parameter in signature.parameters.items() if parameter.annotation is not inspect.Parameter.empty and (not isinstance(parameter.annotation, str))}
    a = ast.parse(dedent(src))
    if len(a.body) != 1 or not isinstance(a.body[0], ast.FunctionDef):
        raise RuntimeError(f'Expected {fn} to be a function')
    f = a.body[0]
    annotation_to_type = {}
    for arg in f.args.args:
        arg_annotation_str = get_annotation_str(arg.annotation) if arg.annotation else None
        if arg_annotation_str is None:
            continue
        arg_name = arg.arg
        if arg_name in name_to_type:
            annotation_to_type[arg_annotation_str] = name_to_type[arg_name]
    literal_return_annotation = get_annotation_str(f.returns)
    valid_literal_annotation = literal_return_annotation is not None
    return_annotation = signature.return_annotation
    valid_return_annotation_type = return_annotation is not inspect.Parameter.empty and (not isinstance(return_annotation, str))
    if valid_literal_annotation and valid_return_annotation_type:
        annotation_to_type[literal_return_annotation] = return_annotation
    return annotation_to_type