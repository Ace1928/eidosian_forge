from __future__ import annotations
import functools
import inspect
import traceback
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import _infra, formatter
@_beartype.beartype
def function_state(fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Mapping[str, Any]:
    bind = inspect.signature(fn).bind(*args, **kwargs)
    return bind.arguments