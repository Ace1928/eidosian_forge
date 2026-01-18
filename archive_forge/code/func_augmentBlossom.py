from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def augmentBlossom(b, v):

    def _recurse(b, v):
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        if isinstance(t, Blossom):
            yield (t, v)
        i = j = b.childs.index(t)
        if i & 1:
            j -= len(b.childs)
            jstep = 1
        else:
            jstep = -1
        while j != 0:
            j += jstep
            t = b.childs[j]
            if jstep == 1:
                w, x = b.edges[j]
            else:
                x, w = b.edges[j - 1]
            if isinstance(t, Blossom):
                yield (t, w)
            j += jstep
            t = b.childs[j]
            if isinstance(t, Blossom):
                yield (t, x)
            mate[w] = x
            mate[x] = w
        b.childs = b.childs[i:] + b.childs[:i]
        b.edges = b.edges[i:] + b.edges[:i]
        blossombase[b] = blossombase[b.childs[0]]
        assert blossombase[b] == v
    stack = [_recurse(b, v)]
    while stack:
        top = stack[-1]
        for args in top:
            stack.append(_recurse(*args))
            break
        else:
            stack.pop()