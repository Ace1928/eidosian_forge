from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def best_model2_alignment(self, sentence_pair, j_pegged=None, i_pegged=0):
    """
        Finds the best alignment according to IBM Model 2

        Used as a starting point for hill climbing in Models 3 and
        above, because it is easier to compute than the best alignments
        in higher models

        :param sentence_pair: Source and target language sentence pair
            to be word-aligned
        :type sentence_pair: AlignedSent

        :param j_pegged: If specified, the alignment point of j_pegged
            will be fixed to i_pegged
        :type j_pegged: int

        :param i_pegged: Alignment point to j_pegged
        :type i_pegged: int
        """
    src_sentence = [None] + sentence_pair.mots
    trg_sentence = ['UNUSED'] + sentence_pair.words
    l = len(src_sentence) - 1
    m = len(trg_sentence) - 1
    alignment = [0] * (m + 1)
    cepts = [[] for i in range(l + 1)]
    for j in range(1, m + 1):
        if j == j_pegged:
            best_i = i_pegged
        else:
            best_i = 0
            max_alignment_prob = IBMModel.MIN_PROB
            t = trg_sentence[j]
            for i in range(0, l + 1):
                s = src_sentence[i]
                alignment_prob = self.translation_table[t][s] * self.alignment_table[i][j][l][m]
                if alignment_prob >= max_alignment_prob:
                    max_alignment_prob = alignment_prob
                    best_i = i
        alignment[j] = best_i
        cepts[best_i].append(j)
    return AlignmentInfo(tuple(alignment), tuple(src_sentence), tuple(trg_sentence), cepts)