from typing import Any, List, Optional, Set, Tuple
import torch
from xformers.ops.common import get_xformers_operator, register_operator
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from xformers.ops.fmha.common import (
@classmethod
def shape_not_supported_reasons(cls, Mq: int, Mkv: int, K: int, Kv: int) -> List[str]:
    reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
    return reasons