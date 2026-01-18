import math
import random
from typing import List, Optional, Sequence, Tuple, Type
import torch
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase
def create_attn_bias(bias_type, batch_size: int, num_heads: int, num_heads_groups: int, q_len: int, kv_len: int, device, dtype, requires_grad: bool, fmt: str, op: Type[AttentionOpBase], page_size: Optional[int]=None):
    if bias_type is None or isinstance(None, bias_type):
        return None
    r = random.Random('-'.join(map(str, [batch_size, q_len, kv_len, dtype, fmt])))
    window_size = {0: 3, 1: 128, 2: 300}[r.randint(0, 2)]
    if bias_type is torch.Tensor:
        if fmt == 'BMK':
            batch_size *= num_heads
            num_heads = 1
        if op in [fmha.small_k.FwOp, fmha.small_k.BwOp]:
            attn_bias = torch.randn((batch_size, num_heads, 1, kv_len), device=device, dtype=dtype) * 3
            attn_bias = attn_bias.expand(batch_size, num_heads, q_len, kv_len)
        else:
            attn_bias = _create_aligned_bias(batch_size, num_heads_groups, num_heads, q_len, kv_len, device=device, dtype=dtype)
            attn_bias[0, 0, 0, :q_len - 1, :kv_len - 1] = -math.inf
            if fmt in ['BMK', 'BMHK']:
                attn_bias = attn_bias[:, 0]
        if requires_grad:
            attn_bias.requires_grad_(True)
        if fmt == 'BMK':
            attn_bias = attn_bias[:, 0]
        return attn_bias
    if bias_type is fmha.attn_bias.LowerTriangularMask:
        return bias_type()
    if bias_type is fmha.attn_bias.LowerTriangularFromBottomRightMask:
        return bias_type()
    if bias_type is fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask:
        return bias_type(window_size)
    if bias_type is fmha.attn_bias.LowerTriangularMaskWithTensorBias:
        attn_bias = _create_aligned_bias(batch_size, num_heads_groups, num_heads, q_len, kv_len, device=device, dtype=dtype)
        if fmt in ['BMK', 'BMHK']:
            attn_bias = attn_bias[:, 0]
        if fmt == 'BMK':
            attn_bias = attn_bias[:, 0]
        if requires_grad:
            attn_bias.requires_grad_(True)
        return fmha.attn_bias.LowerTriangularMaskWithTensorBias(attn_bias)
    if bias_type in [fmha.attn_bias.BlockDiagonalMask, fmha.attn_bias.BlockDiagonalCausalMask, fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask, fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask, fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask]:
        assert fmt in ['BMGHK', 'BMHK']
        max_q_minus_k = None
        if bias_type in {fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask, fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask}:
            max_q_minus_k = 0
        elif bias_type == fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask:
            assert window_size is not None
            max_q_minus_k = window_size - 1
        block_diag = fmha.attn_bias.BlockDiagonalMask.from_seqlens(*_rand_seqlens(r, batch_size, q_len, kv_len, max_q_minus_k=max_q_minus_k))
        if bias_type is fmha.attn_bias.BlockDiagonalCausalMask:
            block_diag = block_diag.make_causal()
        if bias_type in {fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask, fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask}:
            block_diag = fmha.attn_bias.BlockDiagonalMask(q_seqinfo=block_diag.q_seqinfo, k_seqinfo=block_diag.k_seqinfo, _batch_sizes=block_diag._batch_sizes)
            assert window_size is not None
            if bias_type is fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask:
                block_diag = block_diag.make_local_attention(window_size)
            else:
                block_diag = block_diag.make_local_attention_from_bottomright(window_size)
        if bias_type is fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask:
            block_diag = block_diag.make_causal_from_bottomright()
        return block_diag
    if bias_type in [fmha.attn_bias.BlockDiagonalPaddedKeysMask, fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask, fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask, fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask]:
        assert fmt in ['BMHK', 'BMGHK']
        q, k = _rand_seqlens_padded_k(r, batch_size, q_len, kv_len)
        block_diag_type = bias_type._UNPAGED_TYPE if issubclass(bias_type, fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask) else bias_type
        g_block_diag = block_diag_type.from_seqlens(q_seqlen=q, kv_padding=kv_len, kv_seqlen=k)
        if issubclass(bias_type, fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask):
            assert page_size is not None
            pages_per_row = (kv_len + page_size - 1) // page_size
            block_tables = torch.randperm(batch_size * pages_per_row, device=device, dtype=torch.int32).reshape(batch_size, pages_per_row)
            return g_block_diag.make_paged(block_tables=block_tables, page_size=page_size, paged_type=bias_type)
        return g_block_diag
    if bias_type in [fmha.attn_bias.BlockDiagonalCausalWithOffsetGappyKeysMask, fmha.attn_bias.BlockDiagonalGappyKeysMask]:
        assert fmt in ['BMHK', 'BMGHK']
        max_q_minus_k = None if bias_type is fmha.attn_bias.BlockDiagonalGappyKeysMask else 0
        q, k = _rand_seqlens(r, batch_size, q_len, kv_len, max_q_minus_k)
        total_kv_len = kv_len * batch_size
        starts = [r.randint(0, total_kv_len - ki) for ki in k] + [total_kv_len]
        return fmha.attn_bias.BlockDiagonalGappyKeysMask.from_seqlens(q_seqlen=q, kv_seqstarts=starts, kv_seqlen=k)
    if bias_type == fmha.attn_bias.LocalAttentionFromBottomRightMask:
        return bias_type(window_left=r.randint(0, 5), window_right=r.randint(0, 5))
    assert False, f'Unsupported bias type: {bias_type}'