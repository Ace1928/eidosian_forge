from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def binary_precision(preds: Tensor, target: Tensor, threshold: float=0.5, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute `Precision`_ for binary tasks.

    .. math:: \\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}

    Where :math:`\\text{TP}` and :math:`\\text{FP}` represent the number of true positives and
    false positives respecitively.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary {0,1} predictions
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
        If ``multidim_average`` is set to ``global``, the metric returns a scalar value. If ``multidim_average``
        is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar value per sample.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_precision
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> binary_precision(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_precision
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> binary_precision(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import binary_precision
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> binary_precision(preds, target, multidim_average='samplewise')
        tensor([0.4000, 0.0000])

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _precision_recall_reduce('precision', tp, fp, tn, fn, average='binary', multidim_average=multidim_average)