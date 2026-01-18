import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def compute_lookback_includes(self, C, trans, nullable):
    lookdict = {}
    includedict = {}
    dtrans = {}
    for t in trans:
        dtrans[t] = 1
    for state, N in trans:
        lookb = []
        includes = []
        for p in C[state]:
            if p.name != N:
                continue
            lr_index = p.lr_index
            j = state
            while lr_index < p.len - 1:
                lr_index = lr_index + 1
                t = p.prod[lr_index]
                if (j, t) in dtrans:
                    li = lr_index + 1
                    while li < p.len:
                        if p.prod[li] in self.grammar.Terminals:
                            break
                        if p.prod[li] not in nullable:
                            break
                        li = li + 1
                    else:
                        includes.append((j, t))
                g = self.lr0_goto(C[j], t)
                j = self.lr0_cidhash.get(id(g), -1)
            for r in C[j]:
                if r.name != p.name:
                    continue
                if r.len != p.len:
                    continue
                i = 0
                while i < r.lr_index:
                    if r.prod[i] != p.prod[i + 1]:
                        break
                    i = i + 1
                else:
                    lookb.append((j, r))
        for i in includes:
            if i not in includedict:
                includedict[i] = []
            includedict[i].append((state, N))
        lookdict[state, N] = lookb
    return (lookdict, includedict)