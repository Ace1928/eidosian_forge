from typing import Optional, Type, TypeVar
import torch
@classmethod
def make_causal(cls: Type[Self], seq_len: int, to_seq_len: Optional[int]=None, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> Self:
    if not to_seq_len:
        to_seq_len = seq_len
    additive_mask = torch.triu(torch.ones(seq_len, to_seq_len, device=device, dtype=dtype) * float('-inf'), diagonal=1)
    return cls(additive_mask=additive_mask, is_causal=True)