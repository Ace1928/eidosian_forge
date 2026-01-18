import re
from collections import Counter, defaultdict
from nltk.util import ngrams
def corpus_chrf(references, hypotheses, min_len=1, max_len=6, beta=3.0, ignore_whitespace=True):
    """
    Calculates the corpus level CHRF (Character n-gram F-score), it is the
    macro-averaged value of the sentence/segment level CHRF score.

    This implementation of CHRF only supports a single reference at the moment.

        >>> ref1 = str('It is a guide to action that ensures that the military '
        ...            'will forever heed Party commands').split()
        >>> ref2 = str('It is the guiding principle which guarantees the military '
        ...            'forces always being under the command of the Party').split()
        >>>
        >>> hyp1 = str('It is a guide to action which ensures that the military '
        ...            'always obeys the commands of the party').split()
        >>> hyp2 = str('It is to insure the troops forever hearing the activity '
        ...            'guidebook that party direct')
        >>> corpus_chrf([ref1, ref2, ref1, ref2], [hyp1, hyp2, hyp2, hyp1]) # doctest: +ELLIPSIS
        0.3910...

    :param references: a corpus of list of reference sentences, w.r.t. hypotheses
    :type references: list(list(str))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :param beta: the parameter to assign more importance to recall over precision
    :type beta: float
    :param ignore_whitespace: ignore whitespace characters in scoring
    :type ignore_whitespace: bool
    :return: the sentence level CHRF score.
    :rtype: float
    """
    assert len(references) == len(hypotheses), 'The number of hypotheses and their references should be the same'
    num_sents = len(hypotheses)
    ngram_fscores = defaultdict(lambda: list())
    for reference, hypothesis in zip(references, hypotheses):
        reference = _preprocess(reference, ignore_whitespace)
        hypothesis = _preprocess(hypothesis, ignore_whitespace)
        for n in range(min_len, max_len + 1):
            prec, rec, fscore, tp = chrf_precision_recall_fscore_support(reference, hypothesis, n, beta=beta)
            ngram_fscores[n].append(fscore)
    num_ngram_sizes = len(ngram_fscores)
    total_scores = [sum(fscores) for n, fscores in ngram_fscores.items()]
    return sum(total_scores) / num_ngram_sizes / num_sents