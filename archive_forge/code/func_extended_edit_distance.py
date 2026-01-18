import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def extended_edit_distance(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], language: Literal['en', 'ja']='en', return_sentence_level_score: bool=False, alpha: float=2.0, rho: float=0.3, deletion: float=0.2, insertion: float=1.0) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute extended edit distance score (`ExtendedEditDistance`_) [1] for strings or list of strings.

    The metric utilises the Levenshtein distance and extends it by adding a jump operation.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        return_sentence_level_score: An indication of whether sentence-level EED score is to be returned.
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        Extended edit distance score as a tensor

    Example:
        >>> from torchmetrics.functional.text import extended_edit_distance
        >>> preds = ["this is the prediction", "here is an other sample"]
        >>> target = ["this is the reference", "here is another one"]
        >>> extended_edit_distance(preds=preds, target=target)
        tensor(0.3078)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”,
        submitted to WMT 2019. `ExtendedEditDistance`_

    """
    for param_name, param in zip(['alpha', 'rho', 'deletion', 'insertion'], [alpha, rho, deletion, insertion]):
        if not isinstance(param, float) or (isinstance(param, float) and param < 0):
            raise ValueError(f'Parameter `{param_name}` is expected to be a non-negative float.')
    sentence_level_scores = _eed_update(preds, target, language, alpha, rho, deletion, insertion)
    average = _eed_compute(sentence_level_scores)
    if return_sentence_level_score:
        return (average, stack(sentence_level_scores))
    return average