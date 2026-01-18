import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def render_arg_type(annotation) -> str:
    if get_origin(annotation) is Union:
        inner_types = [t for t in get_args(annotation) if t is not type(None)]
        if len(inner_types) == 1:
            return f'{render_arg_type(inner_types[0])}?'
    if get_origin(annotation) is list:
        inner_type, = get_args(annotation)
        return f'{render_arg_type(inner_type)}[]'
    if get_origin(annotation) is tuple:
        return '(' + ', '.join([render_arg_type(t) for t in get_args(annotation)]) + ')'
    if get_origin(annotation) is Annotated:
        inner_type, annotation = get_args(annotation)
        if isinstance(annotation, Alias):
            alias = annotation.name + ('!' if annotation.write else '')
            return f'{render_arg_type(inner_type)}({alias})'
    if annotation is torch.Tensor:
        return 'Tensor'
    if annotation is bool:
        return 'bool'
    if annotation is int:
        return 'int'
    if annotation is float:
        return 'float'
    if annotation is torch.dtype:
        return 'ScalarType'
    if annotation is torch.distributed.ProcessGroup:
        return '__torch__.torch.classes.c10d.ProcessGroup'
    assert False, f'Unable to parse annotation: `{annotation}`'