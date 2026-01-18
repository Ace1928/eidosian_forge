from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def previous_cept(self, j):
    """
        :return: The previous cept of ``j``, or None if ``j`` belongs to
            the first cept
        """
    i = self.alignment[j]
    if i == 0:
        raise ValueError('Words aligned to NULL cannot have a previous cept because NULL has no position')
    previous_cept = i - 1
    while previous_cept > 0 and self.fertility_of_i(previous_cept) == 0:
        previous_cept -= 1
    if previous_cept <= 0:
        previous_cept = None
    return previous_cept