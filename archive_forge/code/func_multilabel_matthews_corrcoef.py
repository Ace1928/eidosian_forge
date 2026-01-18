from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTask
def multilabel_matthews_corrcoef(preds: Tensor, target: Tensor, num_labels: int, threshold: float=0.5, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Calculate `Matthews correlation coefficient`_ for multilabel tasks.

    This metric measures the general correlation or quality of a classification.

    Accepts the following input tensors:

        - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
          [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
          we convert to int tensor with thresholding using the value in ``threshold``.
        - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_matthews_corrcoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_matthews_corrcoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)

    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index, normalize=None)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _matthews_corrcoef_reduce(confmat)