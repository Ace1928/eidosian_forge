import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def pretty_format(self, width=None):
    """
        Return a pretty-printed string representation of this chart.

        :param width: The number of characters allotted to each
            index in the sentence.
        :rtype: str
        """
    if width is None:
        width = 50 // (self.num_leaves() + 1)
    edges = sorted(((e.length(), e.start(), e) for e in self))
    edges = [e for _, _, e in edges]
    return self.pretty_format_leaves(width) + '\n' + '\n'.join((self.pretty_format_edge(edge, width) for edge in edges))