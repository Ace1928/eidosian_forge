import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
@classmethod
def from_words(cls, words, window_size=4):
    if window_size < 4:
        raise ValueError('Specify window_size at least 4')
    ixxx = FreqDist()
    iiii = FreqDist()
    ii = FreqDist()
    iii = FreqDist()
    ixi = FreqDist()
    ixxi = FreqDist()
    iixi = FreqDist()
    ixii = FreqDist()
    for window in ngrams(words, window_size, pad_right=True):
        w1 = window[0]
        if w1 is None:
            continue
        for w2, w3, w4 in _itertools.combinations(window[1:], 3):
            ixxx[w1] += 1
            if w2 is None:
                continue
            ii[w1, w2] += 1
            if w3 is None:
                continue
            iii[w1, w2, w3] += 1
            ixi[w1, w3] += 1
            if w4 is None:
                continue
            iiii[w1, w2, w3, w4] += 1
            ixxi[w1, w4] += 1
            ixii[w1, w3, w4] += 1
            iixi[w1, w2, w4] += 1
    return cls(ixxx, iiii, ii, iii, ixi, ixxi, iixi, ixii)