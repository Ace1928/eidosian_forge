from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def merge_attentions(attn_split: torch.Tensor, lse_split: torch.Tensor, write_lse: bool=True, output_dtype: Optional[torch.dtype]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Combine attention output computed on different parts of K/V for the same
    query to get attention on the whole K/V. See https://arxiv.org/abs/2402.05099
    The result is equal to
        Out_full = (Out1 * exp(LSE1) + Out2 * exp(LSE2) + ...) / (exp(LSE1) + exp(LSE2) + ...)
        LSE_full = log(exp(LSE1) + exp(LSE2) + ...)
    Attention inputs are in BH(G)MK format, stacked along dim 0. Attention output also is in BH(G)MK.

    Args:
        attn_split: [split_k, B, M, G, H, Kq] or [split_k, B, M, H, Kq]
        lse_split: [split_k, B, G, H, M] or [split_k, B, H, M]
        out_dype: dtype of attn_out

    Returns:
        attn_out: [B, M, G, H, Kq] or [B, M, H, Kq]
        lse_out: [B, G, H, M] or [B, H, M] if write_lse
                 or None otherwise
    """
    assert attn_split.ndim == lse_split.ndim + 1, f'attn_split.shape={attn_split.shape!r} lse_split.shape={lse_split.shape!r}'
    is_bmhk = attn_split.ndim == 5
    if is_bmhk:
        attn_split = attn_split.unsqueeze(3)
        lse_split = lse_split.unsqueeze(2)
    split_k, B, M, G, H, Kq = attn_split.shape
    split_k1, B1, G1, H1, M1 = lse_split.shape
    assert B == B1 and G == G1 and (H == H1) and (split_k == split_k1) and (M == M), f'attn_split.shape={attn_split.shape!r} lse_split.shape={lse_split.shape!r} {B}/{B1}, {G}/{G1}, {H}/{H1}, {split_k}/{split_k1}, {M}/{M}'
    attn_split = attn_split.permute(1, 3, 4, 0, 2, 5)
    lse_split = lse_split.permute(1, 2, 3, 0, 4)
    attn_out = torch.empty(B, M, G, H, Kq, device=attn_split.device, dtype=attn_split.dtype if output_dtype is None else output_dtype)
    if write_lse:
        lse_out = torch.empty(B, G, H, M, device=attn_split.device, dtype=lse_split.dtype)
    else:
        lse_out = None
    triton_splitk.merge_attentions(attn_out, lse_out, attn_split, lse_split)
    if is_bmhk:
        attn_out = attn_out[:, :, 0]
        if lse_out is not None:
            lse_out = lse_out[:, 0]
    return (attn_out, lse_out)