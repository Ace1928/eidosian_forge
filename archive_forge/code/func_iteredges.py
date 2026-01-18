import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def iteredges(self):
    """
        Return an iterator over the edges in this chart.  It is
        not guaranteed that new edges which are added to the
        chart before the iterator is exhausted will also be generated.

        :rtype: iter(EdgeI)
        :see: ``edges``, ``select``
        """
    return iter(self._edges)