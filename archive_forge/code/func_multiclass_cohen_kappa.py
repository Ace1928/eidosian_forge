from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def multiclass_cohen_kappa(preds: Tensor, target: Tensor, num_classes: int, weights: Optional[Literal['linear', 'quadratic', 'none']]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Calculate `Cohen's kappa score`_ that measures inter-annotator agreement for multiclass tasks.

    .. math::
        \\kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting


        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_cohen_kappa
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_cohen_kappa(preds, target, num_classes=3)
        tensor(0.6364)

    Example (pred is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_cohen_kappa
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_cohen_kappa(preds, target, num_classes=3)
        tensor(0.6364)

    """
    if validate_args:
        _multiclass_cohen_kappa_arg_validation(num_classes, ignore_index, weights)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _cohen_kappa_reduce(confmat, weights)