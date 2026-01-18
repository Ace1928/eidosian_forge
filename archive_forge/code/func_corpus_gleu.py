from collections import Counter
from nltk.util import everygrams, ngrams
def corpus_gleu(list_of_references, hypotheses, min_len=1, max_len=4):
    """
    Calculate a single corpus-level GLEU score (aka. system-level GLEU) for all
    the hypotheses and their respective references.

    Instead of averaging the sentence level GLEU scores (i.e. macro-average
    precision), Wu et al. (2016) sum up the matching tokens and the max of
    hypothesis and reference tokens for each sentence, then compute using the
    aggregate values.

    From Mike Schuster (via email):
        "For the corpus, we just add up the two statistics n_match and
         n_all = max(n_all_output, n_all_target) for all sentences, then
         calculate gleu_score = n_match / n_all, so it is not just a mean of
         the sentence gleu scores (in our case, longer sentences count more,
         which I think makes sense as they are more difficult to translate)."

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> corpus_gleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
    0.5673...

    The example below show that corpus_gleu() is different from averaging
    sentence_gleu() for hypotheses

    >>> score1 = sentence_gleu([ref1a], hyp1)
    >>> score2 = sentence_gleu([ref2a], hyp2)
    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS
    0.6144...

    :param list_of_references: a list of reference sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :return: The corpus-level GLEU score.
    :rtype: float
    """
    assert len(list_of_references) == len(hypotheses), 'The number of hypotheses and their reference(s) should be the same'
    corpus_n_match = 0
    corpus_n_all = 0
    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_ngrams = Counter(everygrams(hypothesis, min_len, max_len))
        tpfp = sum(hyp_ngrams.values())
        hyp_counts = []
        for reference in references:
            ref_ngrams = Counter(everygrams(reference, min_len, max_len))
            tpfn = sum(ref_ngrams.values())
            overlap_ngrams = ref_ngrams & hyp_ngrams
            tp = sum(overlap_ngrams.values())
            n_all = max(tpfp, tpfn)
            if n_all > 0:
                hyp_counts.append((tp, n_all))
        if hyp_counts:
            n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
            corpus_n_match += n_match
            corpus_n_all += n_all
    if corpus_n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = corpus_n_match / corpus_n_all
    return gleu_score