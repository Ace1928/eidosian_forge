import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def dump_nfa(self, name: str, start: 'NFAState', finish: 'NFAState') -> None:
    print('Dump of NFA for', name)
    todo = [start]
    for i, state in enumerate(todo):
        print('  State', i, state is finish and '(final)' or '')
        for label, next in state.arcs:
            if next in todo:
                j = todo.index(next)
            else:
                j = len(todo)
                todo.append(next)
            if label is None:
                print('    -> %d' % j)
            else:
                print('    %s -> %d' % (label, j))