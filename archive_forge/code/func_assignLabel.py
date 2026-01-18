from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def assignLabel(w, t, v):
    b = inblossom[w]
    assert label.get(w) is None and label.get(b) is None
    label[w] = label[b] = t
    if v is not None:
        labeledge[w] = labeledge[b] = (v, w)
    else:
        labeledge[w] = labeledge[b] = None
    bestedge[w] = bestedge[b] = None
    if t == 1:
        if isinstance(b, Blossom):
            queue.extend(b.leaves())
        else:
            queue.append(b)
    elif t == 2:
        base = blossombase[b]
        assignLabel(mate[base], 1, base)