from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def demographic_parity(preds: torch.Tensor, groups: torch.Tensor, threshold: float=0.5, ignore_index: Optional[int]=None, validate_args: bool=True) -> Dict[str, torch.Tensor]:
    """`Demographic parity`_ compares the positivity rates between all groups.

    If more than two groups are present, the disparity between the lowest and highest group is reported. The lowest
    positivity rate is divided by the highest, so a lower value means more discrimination against the numerator.
    In the results this is also indicated as the key of dict is DP_{identifier_low_group}_{identifier_high_group}.

    .. math::
        \\text{DP} = \\dfrac{\\min_a PR_a}{\\max_a PR_a}.

    where :math:`\\text{PR}` represents the positivity rate for group :math:`\\text{a}`.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
    - ``target`` (int tensor): ``(N, ...)``.

    The additional dimensions are flatted along the batch dimension.

    Args:
        preds: Tensor with predictions.
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a dict where the key identifies the group with the lowest and highest positivity rates
        as follows: DP_{identifier_low_group}_{identifier_high_group}. The value is a tensor with the DP rate.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import demographic_parity
        >>> preds = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> demographic_parity(preds, groups)
        {'DP_0_1': tensor(0.)}

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import demographic_parity
        >>> preds = torch.tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> demographic_parity(preds, groups)
        {'DP_0_1': tensor(0.)}

    """
    num_groups = torch.unique(groups).shape[0]
    target = torch.zeros(preds.shape)
    group_stats = _binary_groups_stat_scores(preds, target, groups, num_groups, threshold, ignore_index, validate_args)
    transformed_group_stats = _groups_stat_transform(group_stats)
    return _compute_binary_demographic_parity(**transformed_group_stats)