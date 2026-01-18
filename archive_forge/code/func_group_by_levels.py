import networkx as nx
from networkx.utils.decorators import not_implemented_for
def group_by_levels(levels):
    L = {}
    for n, lev in levels.items():
        if lev not in L:
            L[lev] = []
        L[lev].append(n)
    return L