import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def num_edges(self):
    """
        Return the number of edges contained in this chart.

        :rtype: int
        """
    return len(self._edge_to_cpls)