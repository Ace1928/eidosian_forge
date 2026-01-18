import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def standard_nonterm_parser(string, pos):
    m = _STANDARD_NONTERM_RE.match(string, pos)
    if not m:
        raise ValueError('Expected a nonterminal, found: ' + string[pos:])
    return (Nonterminal(m.group(1)), m.end())