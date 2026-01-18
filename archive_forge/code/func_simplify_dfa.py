import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def simplify_dfa(self, dfa: List['DFAState']) -> None:
    changes = True
    while changes:
        changes = False
        for i, state_i in enumerate(dfa):
            for j in range(i + 1, len(dfa)):
                state_j = dfa[j]
                if state_i == state_j:
                    del dfa[j]
                    for state in dfa:
                        state.unifystate(state_j, state_i)
                    changes = True
                    break