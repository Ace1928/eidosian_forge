from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def circuit(thisnode, startnode, component):
    closed = False
    path.append(thisnode)
    blocked[thisnode] = True
    for nextnode in component[thisnode]:
        if nextnode == startnode:
            result.append(path[:])
            closed = True
        elif not blocked[nextnode]:
            if circuit(nextnode, startnode, component):
                closed = True
    if closed:
        _unblock(thisnode)
    else:
        for nextnode in component[thisnode]:
            if thisnode not in B[nextnode]:
                B[nextnode].append(thisnode)
    path.pop()
    return closed