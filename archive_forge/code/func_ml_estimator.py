import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
def ml_estimator(self, counts):
    """Calculate the maximum likelihood estimator.

        This can calculate maximum likelihoods for both transitions
        and emissions.

        Arguments:
         - counts -- A dictionary of the counts for each item.

        See estimate_params for a description of the formula used for
        calculation.

        """
    all_ordered = sorted(counts)
    ml_estimation = {}
    cur_letter = None
    cur_letter_counts = 0
    for cur_item in all_ordered:
        if cur_item[0] != cur_letter:
            cur_letter = cur_item[0]
            cur_letter_counts = counts[cur_item]
            cur_position = all_ordered.index(cur_item) + 1
            while cur_position < len(all_ordered) and all_ordered[cur_position][0] == cur_item[0]:
                cur_letter_counts += counts[all_ordered[cur_position]]
                cur_position += 1
        else:
            pass
        cur_ml = counts[cur_item] / cur_letter_counts
        ml_estimation[cur_item] = cur_ml
    return ml_estimation