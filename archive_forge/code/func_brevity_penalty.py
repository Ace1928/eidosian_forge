import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from nltk.util import ngrams
def brevity_penalty(closest_ref_len, hyp_len):
    """
    Calculate brevity penalty.

    As the modified n-gram precision still has the problem from the short
    length sentence, brevity penalty is used to modify the overall BLEU
    score according to length.

    An example from the paper. There are three references with length 12, 15
    and 17. And a concise hypothesis of the length 12. The brevity penalty is 1.

    >>> reference1 = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
    >>> reference2 = list('aaaaaaaaaaaaaaa')   # i.e. ['a'] * 15
    >>> reference3 = list('aaaaaaaaaaaaaaaaa') # i.e. ['a'] * 17
    >>> hypothesis = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
    >>> references = [reference1, reference2, reference3]
    >>> hyp_len = len(hypothesis)
    >>> closest_ref_len =  closest_ref_length(references, hyp_len)
    >>> brevity_penalty(closest_ref_len, hyp_len)
    1.0

    In case a hypothesis translation is shorter than the references, penalty is
    applied.

    >>> references = [['a'] * 28, ['a'] * 28]
    >>> hypothesis = ['a'] * 12
    >>> hyp_len = len(hypothesis)
    >>> closest_ref_len =  closest_ref_length(references, hyp_len)
    >>> brevity_penalty(closest_ref_len, hyp_len)
    0.2635971381157267

    The length of the closest reference is used to compute the penalty. If the
    length of a hypothesis is 12, and the reference lengths are 13 and 2, the
    penalty is applied because the hypothesis length (12) is less then the
    closest reference length (13).

    >>> references = [['a'] * 13, ['a'] * 2]
    >>> hypothesis = ['a'] * 12
    >>> hyp_len = len(hypothesis)
    >>> closest_ref_len =  closest_ref_length(references, hyp_len)
    >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
    0.9200...

    The brevity penalty doesn't depend on reference order. More importantly,
    when two reference sentences are at the same distance, the shortest
    reference sentence length is used.

    >>> references = [['a'] * 13, ['a'] * 11]
    >>> hypothesis = ['a'] * 12
    >>> hyp_len = len(hypothesis)
    >>> closest_ref_len =  closest_ref_length(references, hyp_len)
    >>> bp1 = brevity_penalty(closest_ref_len, hyp_len)
    >>> hyp_len = len(hypothesis)
    >>> closest_ref_len =  closest_ref_length(reversed(references), hyp_len)
    >>> bp2 = brevity_penalty(closest_ref_len, hyp_len)
    >>> bp1 == bp2 == 1
    True

    A test example from mteval-v13a.pl (starting from the line 705):

    >>> references = [['a'] * 11, ['a'] * 8]
    >>> hypothesis = ['a'] * 7
    >>> hyp_len = len(hypothesis)
    >>> closest_ref_len =  closest_ref_length(references, hyp_len)
    >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
    0.8668...

    >>> references = [['a'] * 11, ['a'] * 8, ['a'] * 6, ['a'] * 7]
    >>> hypothesis = ['a'] * 7
    >>> hyp_len = len(hypothesis)
    >>> closest_ref_len =  closest_ref_length(references, hyp_len)
    >>> brevity_penalty(closest_ref_len, hyp_len)
    1.0

    :param hyp_len: The length of the hypothesis for a single sentence OR the
        sum of all the hypotheses' lengths for a corpus
    :type hyp_len: int
    :param closest_ref_len: The length of the closest reference for a single
        hypothesis OR the sum of all the closest references for every hypotheses.
    :type closest_ref_len: int
    :return: BLEU's brevity penalty.
    :rtype: float
    """
    if hyp_len > closest_ref_len:
        return 1
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)