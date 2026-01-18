from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.compute import _auc_compute_without_check, _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def multilabel_auroc(preds: Tensor, target: Tensor, num_labels: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for multilabel tasks.

    The AUROC score summarizes the ROC curve into an single number that describes the performance of a model for
    multiple thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with auroc score per class.
        If `average="micro|macro"|"weighted"` then a single scalar is returned.

    Example:
        >>> from torchmetrics.functional.classification import multilabel_auroc
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> multilabel_auroc(preds, target, num_labels=3, average="macro", thresholds=None)
        tensor(0.6528)
        >>> multilabel_auroc(preds, target, num_labels=3, average=None, thresholds=None)
        tensor([0.6250, 0.5000, 0.8333])
        >>> multilabel_auroc(preds, target, num_labels=3, average="macro", thresholds=5)
        tensor(0.6528)
        >>> multilabel_auroc(preds, target, num_labels=3, average=None, thresholds=5)
        tensor([0.6250, 0.5000, 0.8333])

    """
    if validate_args:
        _multilabel_auroc_arg_validation(num_labels, average, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(preds, target, num_labels, thresholds, ignore_index)
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_auroc_compute(state, num_labels, average, thresholds, ignore_index)