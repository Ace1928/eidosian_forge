from collections import defaultdict
import networkx as nx
def make_N_from_L_C(L, C):
    nodes = L.keys()
    colors = C.keys()
    return {(node, color): sum((1 for v in L[node] if v in C[color])) for node in nodes for color in colors}