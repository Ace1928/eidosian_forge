from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
def recall_at_fixed_precision(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass', 'multilabel'], min_precision: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Optional[Tuple[Tensor, Tensor]]:
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for a
    given precision level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_recall_at_fixed_precision`,
    :func:`~torchmetrics.functional.classification.multiclass_recall_at_fixed_precision` and
    :func:`~torchmetrics.functional.classification.multilabel_recall_at_fixed_precision` for the specific details of
    each argument influence and examples.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_recall_at_fixed_precision(preds, target, min_precision, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
        return multiclass_recall_at_fixed_precision(preds, target, num_classes, min_precision, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
        return multilabel_recall_at_fixed_precision(preds, target, num_labels, min_precision, thresholds, ignore_index, validate_args)
    return None