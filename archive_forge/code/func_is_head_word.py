from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def is_head_word(self, j):
    """
        :return: Whether the word in position ``j`` of the target
            sentence is a head word
        """
    i = self.alignment[j]
    return self.cepts[i][0] == j