from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
def digraph(X, R, G):
    F = {}
    S = []
    N = dict.fromkeys(X, 0)
    for x in X:
        if N[x] == 0:
            traverse(x, S, N, X, R, G, F)
    return F