import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def make_dfa(self, start: 'NFAState', finish: 'NFAState') -> List['DFAState']:
    assert isinstance(start, NFAState)
    assert isinstance(finish, NFAState)

    def closure(state: NFAState) -> Dict[NFAState, int]:
        base: Dict[NFAState, int] = {}
        addclosure(state, base)
        return base

    def addclosure(state: NFAState, base: Dict[NFAState, int]) -> None:
        assert isinstance(state, NFAState)
        if state in base:
            return
        base[state] = 1
        for label, next in state.arcs:
            if label is None:
                addclosure(next, base)
    states = [DFAState(closure(start), finish)]
    for state in states:
        arcs: Dict[str, Dict[NFAState, int]] = {}
        for nfastate in state.nfaset:
            for label, next in nfastate.arcs:
                if label is not None:
                    addclosure(next, arcs.setdefault(label, {}))
        for label, nfaset in sorted(arcs.items()):
            for st in states:
                if st.nfaset == nfaset:
                    break
            else:
                st = DFAState(nfaset, finish)
                states.append(st)
            state.addarc(st, label)
    return states