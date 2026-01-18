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
def binary_auroc(preds: Tensor, target: Tensor, max_fpr: Optional[float]=None, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for binary tasks.

    The AUROC score summarizes the ROC curve into an single number that describes the performance of a model for
    multiple thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        max_fpr: If not ``None``, calculates standardized partial AUC over the range ``[0, max_fpr]``.
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
        A single scalar with the auroc score

    Example:
        >>> from torchmetrics.functional.classification import binary_auroc
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> binary_auroc(preds, target, thresholds=None)
        tensor(0.5000)
        >>> binary_auroc(preds, target, thresholds=5)
        tensor(0.5000)

    """
    if validate_args:
        _binary_auroc_arg_validation(max_fpr, thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(preds, target, thresholds, ignore_index)
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_auroc_compute(state, thresholds, max_fpr)