import flash_attn_cuda
import torch
import torch.nn as nn
def flash_blocksparse_attn_func(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale=None, causal=False, return_attn_probs=False, convert_mask=True):
    """dropout_p should be set to 0.0 during evaluation"""
    func = FlashBlocksparseAttnFun if not return_attn_probs else FlashBlocksparseAttnFunWithS
    if convert_mask:
        blockmask = convert_blockmask(blockmask, causal=causal)
    return func.apply(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal)