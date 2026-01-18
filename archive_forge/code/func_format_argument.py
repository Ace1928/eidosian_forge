from __future__ import annotations
import dataclasses
import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from torch._logging import LazyString
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import sarif
@_beartype.beartype
def format_argument(obj: Any) -> str:
    return f'{type(obj)}'