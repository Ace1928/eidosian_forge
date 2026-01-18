import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def is_binarised(self):
    """
        Return True if all productions are at most binary.
        Note that there can still be empty and unary productions.
        """
    return self._max_len <= 2