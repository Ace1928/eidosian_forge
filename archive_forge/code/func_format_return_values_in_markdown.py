from __future__ import annotations
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils
@_beartype.beartype
def format_return_values_in_markdown(return_values: Any, format_argument: Callable[[Any], str]=formatter.format_argument) -> str:
    return f'{format_argument(return_values)}'