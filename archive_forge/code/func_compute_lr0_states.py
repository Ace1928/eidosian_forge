from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
def compute_lr0_states(self) -> None:
    self.lr0_itemsets = set()
    cache: Dict['State', LR0ItemSet] = {}

    def step(state: LR0ItemSet) -> Iterator[LR0ItemSet]:
        _, unsat = classify_bool(state.closure, lambda rp: rp.is_satisfied)
        d = classify(unsat, lambda rp: rp.next)
        for sym, rps in d.items():
            kernel = fzset({rp.advance(sym) for rp in rps})
            new_state = cache.get(kernel, None)
            if new_state is None:
                closure = set(kernel)
                for rp in kernel:
                    if not rp.is_satisfied and (not rp.next.is_term):
                        closure |= self.expand_rule(rp.next, self.lr0_rules_by_origin)
                new_state = LR0ItemSet(kernel, closure)
                cache[kernel] = new_state
            state.transitions[sym] = new_state
            yield new_state
        self.lr0_itemsets.add(state)
    for _ in bfs(self.lr0_start_states.values(), step):
        pass