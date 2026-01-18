import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def find_nonterminal_transitions(self, C):
    trans = []
    for stateno, state in enumerate(C):
        for p in state:
            if p.lr_index < p.len - 1:
                t = (stateno, p.prod[p.lr_index + 1])
                if t[1] in self.grammar.Nonterminals:
                    if t not in trans:
                        trans.append(t)
    return trans