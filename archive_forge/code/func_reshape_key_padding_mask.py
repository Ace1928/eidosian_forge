from typing import Optional
import torch
def reshape_key_padding_mask(key_padding_mask: torch.Tensor, batched_dim: int) -> torch.Tensor:
    assert key_padding_mask.ndim == 2
    batch_size, src_len = key_padding_mask.size()
    num_heads = batched_dim // batch_size
    return _reshape_key_padding_mask(key_padding_mask, batch_size, src_len, num_heads)