import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def leftcorners(self, cat):
    """
        Return the set of all words that the given category can start with.
        Also called the "first set" in compiler construction.
        """
    raise NotImplementedError('Not implemented yet')