import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from nltk.util import ngrams
def method0(self, p_n, *args, **kwargs):
    """
        No smoothing.
        """
    p_n_new = []
    for i, p_i in enumerate(p_n):
        if p_i.numerator != 0:
            p_n_new.append(p_i)
        else:
            _msg = str('\nThe hypothesis contains 0 counts of {}-gram overlaps.\nTherefore the BLEU score evaluates to 0, independently of\nhow many N-gram overlaps of lower order it contains.\nConsider using lower n-gram order or use SmoothingFunction()').format(i + 1)
            warnings.warn(_msg)
            p_n_new.append(sys.float_info.min)
    return p_n_new