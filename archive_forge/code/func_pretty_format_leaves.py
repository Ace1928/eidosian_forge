import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def pretty_format_leaves(self, width=None):
    """
        Return a pretty-printed string representation of this
        chart's leaves.  This string can be used as a header
        for calls to ``pretty_format_edge``.
        """
    if width is None:
        width = 50 // (self.num_leaves() + 1)
    if self._tokens is not None and width > 1:
        header = '|.'
        for tok in self._tokens:
            header += tok[:width - 1].center(width - 1) + '.'
        header += '|'
    else:
        header = ''
    return header