import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def move_dot_forward(self, new_end):
    """
        Return a new ``TreeEdge`` formed from this edge.
        The new edge's dot position is increased by ``1``,
        and its end index will be replaced by ``new_end``.

        :param new_end: The new end index.
        :type new_end: int
        :rtype: TreeEdge
        """
    return TreeEdge(span=(self._span[0], new_end), lhs=self._lhs, rhs=self._rhs, dot=self._dot + 1)