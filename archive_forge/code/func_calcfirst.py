import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def calcfirst(self, name: str) -> None:
    dfa = self.dfas[name]
    self.first[name] = None
    state = dfa[0]
    totalset: Dict[str, int] = {}
    overlapcheck = {}
    for label in state.arcs:
        if label in self.dfas:
            if label in self.first:
                fset = self.first[label]
                if fset is None:
                    raise ValueError('recursion for rule %r' % name)
            else:
                self.calcfirst(label)
                fset = self.first[label]
                assert fset is not None
            totalset.update(fset)
            overlapcheck[label] = fset
        else:
            totalset[label] = 1
            overlapcheck[label] = {label: 1}
    inverse: Dict[str, str] = {}
    for label, itsfirst in overlapcheck.items():
        for symbol in itsfirst:
            if symbol in inverse:
                raise ValueError('rule %s is ambiguous; %s is in the first sets of %s as well as %s' % (name, symbol, label, inverse[symbol]))
            inverse[symbol] = label
    self.first[name] = totalset