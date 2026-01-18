from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def expandBlossom(b, endstage):

    def _recurse(b, endstage):
        for s in b.childs:
            blossomparent[s] = None
            if isinstance(s, Blossom):
                if endstage and blossomdual[s] == 0:
                    yield s
                else:
                    for v in s.leaves():
                        inblossom[v] = s
            else:
                inblossom[s] = s
        if not endstage and label.get(b) == 2:
            entrychild = inblossom[labeledge[b][1]]
            j = b.childs.index(entrychild)
            if j & 1:
                j -= len(b.childs)
                jstep = 1
            else:
                jstep = -1
            v, w = labeledge[b]
            while j != 0:
                if jstep == 1:
                    p, q = b.edges[j]
                else:
                    q, p = b.edges[j - 1]
                label[w] = None
                label[q] = None
                assignLabel(w, 2, v)
                allowedge[p, q] = allowedge[q, p] = True
                j += jstep
                if jstep == 1:
                    v, w = b.edges[j]
                else:
                    w, v = b.edges[j - 1]
                allowedge[v, w] = allowedge[w, v] = True
                j += jstep
            bw = b.childs[j]
            label[w] = label[bw] = 2
            labeledge[w] = labeledge[bw] = (v, w)
            bestedge[bw] = None
            j += jstep
            while b.childs[j] != entrychild:
                bv = b.childs[j]
                if label.get(bv) == 1:
                    j += jstep
                    continue
                if isinstance(bv, Blossom):
                    for v in bv.leaves():
                        if label.get(v):
                            break
                else:
                    v = bv
                if label.get(v):
                    assert label[v] == 2
                    assert inblossom[v] == bv
                    label[v] = None
                    label[mate[blossombase[bv]]] = None
                    assignLabel(v, 2, labeledge[v][0])
                j += jstep
        label.pop(b, None)
        labeledge.pop(b, None)
        bestedge.pop(b, None)
        del blossomparent[b]
        del blossombase[b]
        del blossomdual[b]
    stack = [_recurse(b, endstage)]
    while stack:
        top = stack[-1]
        for s in top:
            stack.append(_recurse(s, endstage))
            break
        else:
            stack.pop()