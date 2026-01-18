from typing import Dict, Optional, Set
import torch
from torch._ops import OpOverload, OpOverloadPacket, HigherOrderOperator
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase
def is_view_op(schema: torch._C.FunctionSchema) -> bool:
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return alias_info is not None and (not alias_info.is_write)