import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from nltk.util import ngrams
def method7(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
    """
        Smoothing method 7:
        Interpolates methods 4 and 5.
        """
    hyp_len = hyp_len if hyp_len else len(hypothesis)
    p_n = self.method4(p_n, references, hypothesis, hyp_len)
    p_n = self.method5(p_n, references, hypothesis, hyp_len)
    return p_n