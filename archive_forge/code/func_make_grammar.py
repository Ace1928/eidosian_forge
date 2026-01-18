import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def make_grammar(self) -> PgenGrammar:
    c = PgenGrammar()
    names = list(self.dfas.keys())
    names.sort()
    names.remove(self.startsymbol)
    names.insert(0, self.startsymbol)
    for name in names:
        i = 256 + len(c.symbol2number)
        c.symbol2number[name] = i
        c.number2symbol[i] = name
    for name in names:
        dfa = self.dfas[name]
        states = []
        for state in dfa:
            arcs = []
            for label, next in sorted(state.arcs.items()):
                arcs.append((self.make_label(c, label), dfa.index(next)))
            if state.isfinal:
                arcs.append((0, dfa.index(state)))
            states.append(arcs)
        c.states.append(states)
        c.dfas[c.symbol2number[name]] = (states, self.make_first(c, name))
    c.start = c.symbol2number[self.startsymbol]
    return c