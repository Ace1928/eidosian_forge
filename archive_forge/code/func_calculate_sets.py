from collections import Counter, defaultdict
from typing import List, Dict, Iterator, FrozenSet, Set
from ..utils import bfs, fzset, classify
from ..exceptions import GrammarError
from ..grammar import Rule, Terminal, NonTerminal, Symbol
from ..common import ParserConf
def calculate_sets(rules):
    """Calculate FOLLOW sets.

    Adapted from: http://lara.epfl.ch/w/cc09:algorithm_for_first_and_follow_sets"""
    symbols = {sym for rule in rules for sym in rule.expansion} | {rule.origin for rule in rules}
    NULLABLE = set()
    FIRST = {}
    FOLLOW = {}
    for sym in symbols:
        FIRST[sym] = {sym} if sym.is_term else set()
        FOLLOW[sym] = set()
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if set(rule.expansion) <= NULLABLE:
                if update_set(NULLABLE, {rule.origin}):
                    changed = True
            for i, sym in enumerate(rule.expansion):
                if set(rule.expansion[:i]) <= NULLABLE:
                    if update_set(FIRST[rule.origin], FIRST[sym]):
                        changed = True
                else:
                    break
    changed = True
    while changed:
        changed = False
        for rule in rules:
            for i, sym in enumerate(rule.expansion):
                if i == len(rule.expansion) - 1 or set(rule.expansion[i + 1:]) <= NULLABLE:
                    if update_set(FOLLOW[sym], FOLLOW[rule.origin]):
                        changed = True
                for j in range(i + 1, len(rule.expansion)):
                    if set(rule.expansion[i + 1:j]) <= NULLABLE:
                        if update_set(FOLLOW[sym], FIRST[rule.expansion[j]]):
                            changed = True
    return (FIRST, FOLLOW, NULLABLE)