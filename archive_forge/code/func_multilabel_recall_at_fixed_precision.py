from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
def multilabel_recall_at_fixed_precision(preds: Tensor, target: Tensor, num_labels: int, min_precision: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tuple[Tensor, Tensor]:
    """Compute the highest possible recall value given the minimum precision thresholds provided for multilabel tasks.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for a
    given precision level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None`` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~torch.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 2 tensors or 2 lists containing

        - recall: an 1d tensor of size (n_classes, ) with the maximum recall for the given precision level per class
        - thresholds: an 1d tensor of size (n_classes, ) with the corresponding threshold level per class

    Example:
        >>> from torchmetrics.functional.classification import multilabel_recall_at_fixed_precision
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> multilabel_recall_at_fixed_precision(preds, target, num_labels=3, min_precision=0.5, thresholds=None)
        (tensor([1., 1., 1.]), tensor([0.0500, 0.5500, 0.0500]))
        >>> multilabel_recall_at_fixed_precision(preds, target, num_labels=3, min_precision=0.5, thresholds=5)
        (tensor([1., 1., 1.]), tensor([0.0000, 0.5000, 0.0000]))

    """
    if validate_args:
        _multilabel_recall_at_fixed_precision_arg_validation(num_labels, min_precision, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(preds, target, num_labels, thresholds, ignore_index)
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_recall_at_fixed_precision_arg_compute(state, num_labels, thresholds, ignore_index, min_precision)