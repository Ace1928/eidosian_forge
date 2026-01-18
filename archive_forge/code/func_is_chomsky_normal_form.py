import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def is_chomsky_normal_form(self):
    """
        Return True if the grammar is of Chomsky Normal Form, i.e. all productions
        are of the form A -> B C, or A -> "s".
        """
    return self.is_flexible_chomsky_normal_form() and self._all_unary_are_lexical