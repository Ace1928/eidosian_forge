from __future__ import annotations
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils
@_beartype.beartype
def format_message_in_text(fn: Callable, *args: Any, **kwargs: Any) -> str:
    return f'{formatter.display_name(fn)}. '