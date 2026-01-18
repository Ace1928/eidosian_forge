import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def lr0_goto(self, I, x):
    g = self.lr_goto_cache.get((id(I), x))
    if g:
        return g
    s = self.lr_goto_cache.get(x)
    if not s:
        s = {}
        self.lr_goto_cache[x] = s
    gs = []
    for p in I:
        n = p.lr_next
        if n and n.lr_before == x:
            s1 = s.get(id(n))
            if not s1:
                s1 = {}
                s[id(n)] = s1
            gs.append(n)
            s = s1
    g = s.get('$end')
    if not g:
        if gs:
            g = self.lr0_closure(gs)
            s['$end'] = g
        else:
            s['$end'] = gs
    self.lr_goto_cache[id(I), x] = g
    return g