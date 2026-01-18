from typing import Optional, Union
import torch
import triton
import triton.language as tl
def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[int]=None, interleaved=False, inplace=False, conjugate=False) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, 'If cu_seqlens is passed in, then max_seqlen must be passed'
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, 'rotary_dim must be <= headdim'
    assert headdim <= 256, 'Only support headdim <= 256'
    assert seqlen_ro >= seqlen, 'seqlen_ro must be >= seqlen'
    assert cos.dtype == sin.dtype, f'cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}'
    assert x.dtype == cos.dtype, f'Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}'
    cos, sin = (cos.contiguous(), sin.contiguous())
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and (not inplace):
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    BLOCK_K = 32 if rotary_dim <= 32 else 64 if rotary_dim <= 64 else 128 if rotary_dim <= 128 else 256
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)
    BLOCK_M = 4 if interleaved else 8 if rotary_dim <= 64 else 4
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](output, x, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128, output.stride(0) if not is_varlen else 0, output.stride(-3), output.stride(-2), output.stride(-1), x.stride(0) if not is_varlen else 0, x.stride(-3), x.stride(-2), x.stride(-1), BLOCK_K, isinstance(seqlen_offsets, torch.Tensor), is_varlen, interleaved, conjugate, BLOCK_M)
    return output