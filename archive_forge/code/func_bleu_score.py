from collections import Counter
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
def bleu_score(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], n_gram: int=4, smooth: bool=False, weights: Optional[Sequence[float]]=None) -> Tensor:
    """Calculate `BLEU score`_ of machine translated text with one or more references.

    Args:
        preds: An iterable of machine translated corpus
        target: An iterable of iterables of reference corpus
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether to apply smoothing - see [2]
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Return:
        Tensor with BLEU Score

    Raises:
        ValueError: If ``preds`` and ``target`` corpus have different lengths.
        ValueError: If a length of a list of weights is not ``None`` and not equal to ``n_gram``.

    Example:
        >>> from torchmetrics.functional.text import bleu_score
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> bleu_score(preds, target)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_

        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_

    """
    preds_ = [preds] if isinstance(preds, str) else preds
    target_ = [[tgt] if isinstance(tgt, str) else tgt for tgt in target]
    if len(preds_) != len(target_):
        raise ValueError(f'Corpus has different size {len(preds_)} != {len(target_)}')
    if weights is not None and len(weights) != n_gram:
        raise ValueError(f'List of weights has different weights than `n_gram`: {len(weights)} != {n_gram}')
    if weights is None:
        weights = [1.0 / n_gram] * n_gram
    numerator = torch.zeros(n_gram)
    denominator = torch.zeros(n_gram)
    preds_len = tensor(0.0)
    target_len = tensor(0.0)
    preds_len, target_len = _bleu_score_update(preds_, target_, numerator, denominator, preds_len, target_len, n_gram, _tokenize_fn)
    return _bleu_score_compute(preds_len, target_len, numerator, denominator, n_gram, weights, smooth)