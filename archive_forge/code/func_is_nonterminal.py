import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def is_nonterminal(item):
    """
    :return: True if the item is a ``Nonterminal``.
    :rtype: bool
    """
    return isinstance(item, Nonterminal)