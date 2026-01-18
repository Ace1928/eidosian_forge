import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from nltk.util import ngrams
def method6(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
    """
        Smoothing method 6:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn−1 will be the same as that between pn−1 and pn−2; from
        Gao and He (2013) Training MRF-Based Phrase Translation Models using
        Gradient Ascent. In NAACL.
        """
    hyp_len = hyp_len if hyp_len else len(hypothesis)
    assert p_n[2], 'This smoothing method requires non-zero precision for bigrams.'
    for i, p_i in enumerate(p_n):
        if i in [0, 1]:
            continue
        else:
            pi0 = 0 if p_n[i - 2] == 0 else p_n[i - 1] ** 2 / p_n[i - 2]
            m = p_i.numerator
            l = sum((1 for _ in ngrams(hypothesis, i + 1)))
            p_n[i] = (m + self.alpha * pi0) / (l + self.alpha)
    return p_n