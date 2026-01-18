from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def neighboring(self, alignment_info, j_pegged=None):
    """
        Determine the neighbors of ``alignment_info``, obtained by
        moving or swapping one alignment point

        :param j_pegged: If specified, neighbors that have a different
            alignment point from j_pegged will not be considered
        :type j_pegged: int

        :return: A set neighboring alignments represented by their
            ``AlignmentInfo``
        :rtype: set(AlignmentInfo)
        """
    neighbors = set()
    l = len(alignment_info.src_sentence) - 1
    m = len(alignment_info.trg_sentence) - 1
    original_alignment = alignment_info.alignment
    original_cepts = alignment_info.cepts
    for j in range(1, m + 1):
        if j != j_pegged:
            for i in range(0, l + 1):
                new_alignment = list(original_alignment)
                new_cepts = deepcopy(original_cepts)
                old_i = original_alignment[j]
                new_alignment[j] = i
                insort_left(new_cepts[i], j)
                new_cepts[old_i].remove(j)
                new_alignment_info = AlignmentInfo(tuple(new_alignment), alignment_info.src_sentence, alignment_info.trg_sentence, new_cepts)
                neighbors.add(new_alignment_info)
    for j in range(1, m + 1):
        if j != j_pegged:
            for other_j in range(1, m + 1):
                if other_j != j_pegged and other_j != j:
                    new_alignment = list(original_alignment)
                    new_cepts = deepcopy(original_cepts)
                    other_i = original_alignment[other_j]
                    i = original_alignment[j]
                    new_alignment[j] = other_i
                    new_alignment[other_j] = i
                    new_cepts[other_i].remove(other_j)
                    insort_left(new_cepts[other_i], j)
                    new_cepts[i].remove(j)
                    insort_left(new_cepts[i], other_j)
                    new_alignment_info = AlignmentInfo(tuple(new_alignment), alignment_info.src_sentence, alignment_info.trg_sentence, new_cepts)
                    neighbors.add(new_alignment_info)
    return neighbors