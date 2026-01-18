import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def parse_inputs(self, schema: torch.FunctionSchema, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    for argument, value in zip_arguments(schema, args, kwargs):
        is_write = argument.alias_info is not None and argument.alias_info.is_write
        pytree.tree_map_(functools.partial(self._handle_argument, is_write=is_write, name=argument.name), value)