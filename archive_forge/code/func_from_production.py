import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
@staticmethod
def from_production(production, index):
    """
        Return a new ``TreeEdge`` formed from the given production.
        The new edge's left-hand side and right-hand side will
        be taken from ``production``; its span will be
        ``(index,index)``; and its dot position will be ``0``.

        :rtype: TreeEdge
        """
    return TreeEdge(span=(index, index), lhs=production.lhs(), rhs=production.rhs(), dot=0)