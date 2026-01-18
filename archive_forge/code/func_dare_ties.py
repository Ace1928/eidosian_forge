import warnings
from typing import List, Literal
import torch
def dare_ties(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float, majority_sign_method: Literal['total', 'frequency']='total') -> torch.Tensor:
    """
    Merge the task tensors using `dare ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='random', rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors