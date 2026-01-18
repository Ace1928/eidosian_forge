from abc import ABCMeta, abstractmethod
from functools import total_ordering
from nltk.internals import raise_unorderable_types
def is_forward(self):
    return self._dir == '/'