from typing import Optional, Type, TypeVar
import torch
def make_crop(self, seq_len: int, to_seq_len: Optional[int]=None) -> 'AttentionMask':
    """
        Return a cropped attention mask, whose underlying tensor is a view of this one
        """
    if not to_seq_len:
        to_seq_len = seq_len
    return AttentionMask(self.values[:, :seq_len, :to_seq_len], is_causal=self.is_causal)