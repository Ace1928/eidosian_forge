import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def cfg_demo():
    """
    A demonstration showing how ``CFGs`` can be created and used.
    """
    from nltk import CFG, Production, nonterminals
    S, NP, VP, PP = nonterminals('S, NP, VP, PP')
    N, V, P, Det = nonterminals('N, V, P, Det')
    VP_slash_NP = VP / NP
    print('Some nonterminals:', [S, NP, VP, PP, N, V, P, Det, VP / NP])
    print('    S.symbol() =>', repr(S.symbol()))
    print()
    print(Production(S, [NP]))
    grammar = CFG.fromstring("\n      S -> NP VP\n      PP -> P NP\n      NP -> Det N | NP PP\n      VP -> V NP | VP PP\n      Det -> 'a' | 'the'\n      N -> 'dog' | 'cat'\n      V -> 'chased' | 'sat'\n      P -> 'on' | 'in'\n    ")
    print('A Grammar:', repr(grammar))
    print('    grammar.start()       =>', repr(grammar.start()))
    print('    grammar.productions() =>', end=' ')
    print(repr(grammar.productions()).replace(',', ',\n' + ' ' * 25))
    print()