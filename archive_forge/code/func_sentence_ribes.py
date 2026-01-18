import math
from itertools import islice
from nltk.util import choose, ngrams
def sentence_ribes(references, hypothesis, alpha=0.25, beta=0.1):
    """
    The RIBES (Rank-based Intuitive Bilingual Evaluation Score) from
    Hideki Isozaki, Tsutomu Hirao, Kevin Duh, Katsuhito Sudoh and
    Hajime Tsukada. 2010. "Automatic Evaluation of Translation Quality for
    Distant Language Pairs". In Proceedings of EMNLP.
    https://www.aclweb.org/anthology/D/D10/D10-1092.pdf

    The generic RIBES scores used in shared task, e.g. Workshop for
    Asian Translation (WAT) uses the following RIBES calculations:

        RIBES = kendall_tau * (alpha**p1) * (beta**bp)

    Please note that this re-implementation differs from the official
    RIBES implementation and though it emulates the results as describe
    in the original paper, there are further optimization implemented
    in the official RIBES script.

    Users are encouraged to use the official RIBES script instead of this
    implementation when evaluating your machine translation system. Refer
    to https://www.kecl.ntt.co.jp/icl/lirg/ribes/ for the official script.

    :param references: a list of reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param alpha: hyperparameter used as a prior for the unigram precision.
    :type alpha: float
    :param beta: hyperparameter used as a prior for the brevity penalty.
    :type beta: float
    :return: The best ribes score from one of the references.
    :rtype: float
    """
    best_ribes = -1.0
    for reference in references:
        worder = word_rank_alignment(reference, hypothesis)
        nkt = kendall_tau(worder)
        bp = min(1.0, math.exp(1.0 - len(reference) / len(hypothesis)))
        p1 = len(worder) / len(hypothesis)
        _ribes = nkt * p1 ** alpha * bp ** beta
        if _ribes > best_ribes:
            best_ribes = _ribes
    return best_ribes