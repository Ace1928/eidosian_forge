from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch
def find_creation_sequence(G):
    """
    Find a threshold subgraph that is close to largest in G.
    Returns the labeled creation sequence of that threshold graph.
    """
    cs = []
    H = G
    while H.order() > 0:
        dsdict = dict(H.degree())
        ds = [(d, v) for v, d in dsdict.items()]
        ds.sort()
        if ds[-1][0] == 0:
            cs.extend(zip(dsdict, ['i'] * (len(ds) - 1) + ['d']))
            break
        while ds[0][0] == 0:
            d, iso = ds.pop(0)
            cs.append((iso, 'i'))
        d, bigv = ds.pop()
        cs.append((bigv, 'd'))
        H = H.subgraph(H.neighbors(bigv))
    cs.reverse()
    return cs