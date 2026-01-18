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
def sarif_to_json(attr_cls_obj: _SarifClass, indent: Optional[str]=' ') -> str:
    dict = dataclasses.asdict(attr_cls_obj)
    dict = _convert_key(dict, snake_case_to_camel_case)
    return json.dumps(dict, indent=indent, separators=(',', ':'))