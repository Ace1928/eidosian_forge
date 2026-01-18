import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def no_such_operator(*args, **kwargs):
    raise RuntimeError(f'No such operator {library}::{name} - did you forget to build xformers with `python setup.py develop`?')