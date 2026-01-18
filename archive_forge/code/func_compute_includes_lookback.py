from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
def compute_includes_lookback(self):
    for nt in self.nonterminal_transitions:
        state, nonterminal = nt
        includes = []
        lookback = self.lookback[nt]
        for rp in state.closure:
            if rp.rule.origin != nonterminal:
                continue
            state2 = state
            for i in range(rp.index, len(rp.rule.expansion)):
                s = rp.rule.expansion[i]
                nt2 = (state2, s)
                state2 = state2.transitions[s]
                if nt2 not in self.reads:
                    continue
                for j in range(i + 1, len(rp.rule.expansion)):
                    if rp.rule.expansion[j] not in self.NULLABLE:
                        break
                else:
                    includes.append(nt2)
            if rp.index == 0:
                for rp2 in state2.closure:
                    if rp2.rule == rp.rule and rp2.is_satisfied:
                        lookback.add((state2, rp2.rule))
        for nt2 in includes:
            self.includes[nt2].add(nt)