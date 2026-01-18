from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def multilabel_stat_scores(preds: Tensor, target: Tensor, num_labels: int, threshold: float=0.5, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute the true positives, false positives, true negatives, false negatives and support for multilabel tasks.

    Related to `Type I and Type II errors`_.

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
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on ``average`` and ``multidim_average`` parameters:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
          - If ``average=None/'none'``, the shape will be ``(C, 5)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
          - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, multidim_average='samplewise', average='micro')
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, multidim_average='samplewise', average=None)
        tensor([[[1, 1, 0, 0, 1],
                 [1, 1, 0, 0, 1],
                 [0, 1, 0, 1, 1]],
                [[0, 0, 0, 2, 2],
                 [0, 2, 0, 0, 0],
                 [0, 0, 1, 1, 1]]])

    """
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _multilabel_stat_scores_compute(tp, fp, tn, fn, average, multidim_average)