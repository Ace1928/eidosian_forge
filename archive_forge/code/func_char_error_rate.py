from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
def char_error_rate(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Compute Character Error Rate used for performance of an automatic speech recognition system.

    This value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better
    the performance of the ASR system with a CER of 0 being a perfect score.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Character error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> char_error_rate(preds=preds, target=target)
        tensor(0.3415)

    """
    errors, total = _cer_update(preds, target)
    return _cer_compute(errors, total)