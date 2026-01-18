import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def nonterminals(symbols):
    """
    Given a string containing a list of symbol names, return a list of
    ``Nonterminals`` constructed from those symbols.

    :param symbols: The symbol name string.  This string can be
        delimited by either spaces or commas.
    :type symbols: str
    :return: A list of ``Nonterminals`` constructed from the symbol
        names given in ``symbols``.  The ``Nonterminals`` are sorted
        in the same order as the symbols names.
    :rtype: list(Nonterminal)
    """
    if ',' in symbols:
        symbol_list = symbols.split(',')
    else:
        symbol_list = symbols.split()
    return [Nonterminal(s.strip()) for s in symbol_list]