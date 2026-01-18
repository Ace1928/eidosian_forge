from typing import Optional, Type, TypeVar
import torch
@classmethod
def from_multiplicative(cls: Type[Self], x: torch.Tensor) -> Self:
    """
        Create an AttentionMask given a multiplicative attention mask.
        """
    assert not x.dtype == torch.bool
    additive_mask = torch.empty_like(x, dtype=torch.float, device=x.device)
    x = x.bool()
    additive_mask.masked_fill_(x, 0.0)
    additive_mask.masked_fill_(~x, float('-inf'))
    return cls(additive_mask)