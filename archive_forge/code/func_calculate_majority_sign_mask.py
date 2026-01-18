import warnings
from typing import List, Literal
import torch
def calculate_majority_sign_mask(tensor: torch.Tensor, method: Literal['total', 'frequency']='total') -> torch.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`torch.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The majority sign mask.
    """
    sign = tensor.sign()
    if method == 'total':
        sign_magnitude = tensor.sum(dim=0)
    elif method == 'frequency':
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign