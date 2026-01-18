from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.recall_fixed_precision import (
from torchmetrics.utilities.enums import ClassificationTask
def precision_at_fixed_recall(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass', 'multilabel'], min_recall: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Optional[Tuple[Tensor, Tensor]]:
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for a
    given precision level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_precision_at_fixed_recall`,
    :func:`~torchmetrics.functional.classification.multiclass_precision_at_fixed_recall` and
    :func:`~torchmetrics.functional.classification.multilabel_precision_at_fixed_recall` for the specific details of
    each argument influence and examples.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_precision_at_fixed_recall(preds, target, min_recall, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
        return multiclass_precision_at_fixed_recall(preds, target, num_classes, min_recall, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
        return multilabel_precision_at_fixed_recall(preds, target, num_labels, min_recall, thresholds, ignore_index, validate_args)
    return None