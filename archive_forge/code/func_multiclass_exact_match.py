from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTaskNoBinary
def multiclass_exact_match(preds: Tensor, target: Tensor, num_classes: int, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute Exact match (also known as subset accuracy) for multiclass tasks.

    Exact Match is a stricter version of accuracy where all labels have to match exactly for the sample to be
    correctly classified.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of labels
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global`` the output will be a scalar tensor
        - If ``multidim_average`` is set to ``samplewise`` the output will be a tensor of shape ``(N,)``

    Example (multidim tensors):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_exact_match
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_exact_match(preds, target, num_classes=3, multidim_average='global')
        tensor(0.5000)

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_exact_match
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_exact_match(preds, target, num_classes=3, multidim_average='samplewise')
        tensor([1., 0.])

    """
    top_k, average = (1, None)
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    correct, total = _multiclass_exact_match_update(preds, target, multidim_average, ignore_index)
    return _exact_match_reduce(correct, total)