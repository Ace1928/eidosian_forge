from typing import Optional, Tuple
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def multiclass_hinge_loss(preds: Tensor, target: Tensor, num_classes: int, squared: bool=False, multiclass_mode: Literal['crammer-singer', 'one-vs-all']='crammer-singer', ignore_index: Optional[int]=None, validate_args: bool=False) -> Tensor:
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used:

    .. math::
        \\text{Hinge loss} = \\max\\left(0, 1 - \\hat{y}_y + \\max_{i \\ne y} (\\hat{y}_i)\\right)

    Where :math:`y \\in {0, ..., \\mathrm{C}}` is the target class (where :math:`\\mathrm{C}` is the number of classes),
    and :math:`\\hat{y} \\in \\mathbb{R}^\\mathrm{C}` is the predicted output per class. Alternatively, the metric can
    also be computed in one-vs-all approach, where each class is valued against all other classes in a binary fashion.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Determines how to compute the metric
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_hinge_loss
        >>> preds = tensor([[0.25, 0.20, 0.55],
        ...                 [0.55, 0.05, 0.40],
        ...                 [0.10, 0.30, 0.60],
        ...                 [0.90, 0.05, 0.05]])
        >>> target = tensor([0, 1, 2, 0])
        >>> multiclass_hinge_loss(preds, target, num_classes=3)
        tensor(0.9125)
        >>> multiclass_hinge_loss(preds, target, num_classes=3, squared=True)
        tensor(1.1131)
        >>> multiclass_hinge_loss(preds, target, num_classes=3, multiclass_mode='one-vs-all')
        tensor([0.8750, 1.1250, 1.1000])

    """
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    measures, total = _multiclass_hinge_loss_update(preds, target, squared, multiclass_mode)
    return _hinge_loss_compute(measures, total)