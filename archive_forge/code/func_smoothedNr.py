import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def smoothedNr(self, r):
    """
        Return the number of samples with count r.

        :param r: The amount of frequency.
        :type r: int
        :rtype: float
        """
    return math.exp(self._intercept + self._slope * math.log(r))