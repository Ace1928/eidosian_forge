import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def set_strategy(self, strategy):
    """
        Change the strategy that the parser uses to decide which edges
        to add to the chart.

        :type strategy: list(ChartRuleI)
        :param strategy: A list of rules that should be used to decide
            what edges to add to the chart.
        """
    if strategy == self._strategy:
        return
    self._strategy = strategy[:]
    self._restart = True