import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def dg_demo():
    """
    A demonstration showing the creation and inspection of a
    ``DependencyGrammar``.
    """
    grammar = DependencyGrammar.fromstring("\n    'scratch' -> 'cats' | 'walls'\n    'walls' -> 'the'\n    'cats' -> 'the'\n    ")
    print(grammar)