import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def reads_relation(self, C, trans, empty):
    rel = []
    state, N = trans
    g = self.lr0_goto(C[state], N)
    j = self.lr0_cidhash.get(id(g), -1)
    for p in g:
        if p.lr_index < p.len - 1:
            a = p.prod[p.lr_index + 1]
            if a in empty:
                rel.append((j, a))
    return rel