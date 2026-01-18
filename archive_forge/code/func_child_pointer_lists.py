import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def child_pointer_lists(self, edge):
    """
        Return the set of child pointer lists for the given edge.
        Each child pointer list is a list of edges that have
        been used to form this edge.

        :rtype: list(list(EdgeI))
        """
    return self._edge_to_cpls.get(edge, {}).keys()