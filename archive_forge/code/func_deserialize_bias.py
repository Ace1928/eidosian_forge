from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
@staticmethod
def deserialize_bias(attn_bias_ctx, attn_bias_tensor: Optional[torch.Tensor]) -> Any:
    if attn_bias_tensor is None:
        return attn_bias_ctx
    return attn_bias_tensor