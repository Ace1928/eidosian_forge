from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def multilabel_accuracy(preds: Tensor, target: Tensor, num_labels: int, threshold: float=0.5, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute `Accuracy`_ for multilabel tasks.

    .. math::
        \\text{Accuracy} = \\frac{1}{N}\\sum_i^N 1(y_i = \\hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a
    tensor of predictions.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

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
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_accuracy(preds, target, num_labels=3)
        tensor(0.6667)
        >>> multilabel_accuracy(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.5000])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_accuracy(preds, target, num_labels=3)
        tensor(0.6667)
        >>> multilabel_accuracy(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.5000])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> multilabel_accuracy(preds, target, num_labels=3, multidim_average='samplewise')
        tensor([0.3333, 0.1667])
        >>> multilabel_accuracy(preds, target, num_labels=3, multidim_average='samplewise', average=None)
        tensor([[0.5000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000]])

    """
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _accuracy_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average, multilabel=True)