import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def logprob(self):
    """
        Return ``log(p)``, where ``p`` is the probability associated
        with this object.

        :rtype: float
        """
    if self.__logprob is None:
        if self.__prob is None:
            return None
        self.__logprob = math.log(self.__prob, 2)
    return self.__logprob