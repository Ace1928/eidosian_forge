import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel4
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def hillclimb(self, alignment_info, j_pegged=None):
    """
        Starting from the alignment in ``alignment_info``, look at
        neighboring alignments iteratively for the best one, according
        to Model 4

        Note that Model 4 scoring is used instead of Model 5 because the
        latter is too expensive to compute.

        There is no guarantee that the best alignment in the alignment
        space will be found, because the algorithm might be stuck in a
        local maximum.

        :param j_pegged: If specified, the search will be constrained to
            alignments where ``j_pegged`` remains unchanged
        :type j_pegged: int

        :return: The best alignment found from hill climbing
        :rtype: AlignmentInfo
        """
    alignment = alignment_info
    max_probability = IBMModel4.model4_prob_t_a_given_s(alignment, self)
    while True:
        old_alignment = alignment
        for neighbor_alignment in self.neighboring(alignment, j_pegged):
            neighbor_probability = IBMModel4.model4_prob_t_a_given_s(neighbor_alignment, self)
            if neighbor_probability > max_probability:
                alignment = neighbor_alignment
                max_probability = neighbor_probability
        if alignment == old_alignment:
            break
    alignment.score = max_probability
    return alignment