from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
@classmethod
def apply_bmhk(cls, inp: Inputs, needs_gradient: bool) -> Tuple[torch.Tensor, Optional[Context]]:
    if type(inp.attn_bias) not in FwOp.SUPPORTED_ATTN_BIAS_TYPES:
        raise NotImplementedError('Unsupported attn_bias type')
    seqstart_k, seqstart_q, max_seqlen_q = _get_seqlen_info(inp)
    out, lse, rng_seed, rng_offset = cls.OPERATOR(query=inp.query, key=inp.key, value=inp.value, attn_bias=_get_tensor_bias(inp.attn_bias), seqstart_q=seqstart_q, seqstart_k=seqstart_k, max_seqlen_q=max_seqlen_q, dropout_p=inp.p, compute_logsumexp=needs_gradient, custom_mask_type=_custom_mask_type(inp.attn_bias), scale=inp.scale, seqlen_k=inp.attn_bias.k_seqinfo.seqlen if isinstance(inp.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask) else None, window_size=inp.attn_bias._window_size if isinstance(inp.attn_bias, (BlockDiagonalCausalLocalAttentionMask, BlockDiagonalCausalLocalAttentionFromBottomRightMask, LowerTriangularFromBottomRightLocalAttentionMask)) else None)
    ctx: Optional[Context] = None
    if needs_gradient:
        ctx = Context(out=out, lse=lse, op_bw=BwOp if inp.p != 0 else None)
        if inp.p != 0:
            ctx.rng_state = torch.tensor([rng_seed, rng_offset], dtype=torch.int64, device='cpu')
    return (out, ctx)