from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def binary_groups_stat_rates(preds: torch.Tensor, target: torch.Tensor, groups: torch.Tensor, num_groups: int, threshold: float=0.5, ignore_index: Optional[int]=None, validate_args: bool=True) -> Dict[str, torch.Tensor]:
    """Compute the true/false positives and true/false negatives rates for binary classification by group.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

    The additional dimensions are flatted along the batch dimension.

    Args:
        preds: Tensor with predictions.
        target: Tensor with true labels.
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
        num_groups: The number of groups.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a dict with a group identifier as key and a tensor with the tp, fp, tn and fn rates as value.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import binary_groups_stat_rates
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> binary_groups_stat_rates(preds, target, groups, 2)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_groups_stat_rates
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> binary_groups_stat_rates(preds, target, groups, 2)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    """
    group_stats = _binary_groups_stat_scores(preds, target, groups, num_groups, threshold, ignore_index, validate_args)
    return _groups_reduce(group_stats)