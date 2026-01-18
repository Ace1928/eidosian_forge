from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def binary_confusion_matrix(preds: Tensor, target: Tensor, threshold: float=0.5, normalize: Optional[Literal['true', 'pred', 'all', 'none']]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute the `confusion matrix`_ for binary tasks.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        A ``[2, 2]`` tensor

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_confusion_matrix
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> binary_confusion_matrix(preds, target)
        tensor([[2, 0],
                [1, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_confusion_matrix
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_confusion_matrix(preds, target)
        tensor([[2, 0],
                [1, 1]])

    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _binary_confusion_matrix_compute(confmat, normalize)