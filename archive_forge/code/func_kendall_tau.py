import math
from itertools import islice
from nltk.util import choose, ngrams
def kendall_tau(worder, normalize=True):
    """
    Calculates the Kendall's Tau correlation coefficient given the *worder*
    list of word alignments from word_rank_alignment(), using the formula:

        tau = 2 * num_increasing_pairs / num_possible_pairs -1

    Note that the no. of increasing pairs can be discontinuous in the *worder*
    list and each each increasing sequence can be tabulated as choose(len(seq), 2)
    no. of increasing pairs, e.g.

        >>> worder = [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]
        >>> number_possible_pairs = choose(len(worder), 2)
        >>> round(kendall_tau(worder, normalize=False),3)
        -0.236
        >>> round(kendall_tau(worder),3)
        0.382

    :param worder: The worder list output from word_rank_alignment
    :type worder: list(int)
    :param normalize: Flag to indicate normalization to between 0.0 and 1.0.
    :type normalize: boolean
    :return: The Kendall's Tau correlation coefficient.
    :rtype: float
    """
    worder_len = len(worder)
    if worder_len < 2:
        tau = -1
    else:
        increasing_sequences = find_increasing_sequences(worder)
        num_increasing_pairs = sum((choose(len(seq), 2) for seq in increasing_sequences))
        num_possible_pairs = choose(worder_len, 2)
        tau = 2 * num_increasing_pairs / num_possible_pairs - 1
    if normalize:
        return (tau + 1) / 2
    else:
        return tau