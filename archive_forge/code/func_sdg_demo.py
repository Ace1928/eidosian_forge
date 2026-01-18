import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def sdg_demo():
    """
    A demonstration of how to read a string representation of
    a CoNLL format dependency tree.
    """
    from nltk.parse import DependencyGraph
    dg = DependencyGraph('\n    1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _\n    2   had               heb               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _\n    3   met               met               Prep  Prep  voor                             8   mod     _  _\n    4   haar              haar              Pron  Pron  bez|3|ev|neut|attr               5   det     _  _\n    5   moeder            moeder            N     N     soort|ev|neut                    3   obj1    _  _\n    6   kunnen            kan               V     V     hulp|ott|1of2of3|mv              2   vc      _  _\n    7   gaan              ga                V     V     hulp|inf                         6   vc      _  _\n    8   winkelen          winkel            V     V     intrans|inf                      11  cnj     _  _\n    9   ,                 ,                 Punc  Punc  komma                            8   punct   _  _\n    10  zwemmen           zwem              V     V     intrans|inf                      11  cnj     _  _\n    11  of                of                Conj  Conj  neven                            7   vc      _  _\n    12  terrassen         terras            N     N     soort|mv|neut                    11  cnj     _  _\n    13  .                 .                 Punc  Punc  punt                             12  punct   _  _\n    ')
    tree = dg.tree()
    print(tree.pprint())