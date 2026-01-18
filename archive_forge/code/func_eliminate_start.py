import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
@classmethod
def eliminate_start(cls, grammar):
    """
        Eliminate start rule in case it appears on RHS
        Example: S -> S0 S1 and S0 -> S1 S
        Then another rule S0_Sigma -> S is added
        """
    start = grammar.start()
    result = []
    need_to_add = None
    for rule in grammar.productions():
        if start in rule.rhs():
            need_to_add = True
        result.append(rule)
    if need_to_add:
        start = Nonterminal('S0_SIGMA')
        result.append(Production(start, [grammar.start()]))
        n_grammar = CFG(start, result)
        return n_grammar
    return grammar