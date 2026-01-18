from typing import Optional, Type, TypeVar
import torch
@classmethod
def from_bool(cls: Type[Self], x: torch.Tensor) -> Self:
    """
        Create an AttentionMask given a boolean pattern.
        .. warning: we assume here that True implies that the value should be computed
        """
    assert x.dtype == torch.bool
    additive_mask = torch.empty_like(x, dtype=torch.float, device=x.device)
    additive_mask.masked_fill_(x, 0.0)
    additive_mask.masked_fill_(~x, float('-inf'))
    return cls(additive_mask)