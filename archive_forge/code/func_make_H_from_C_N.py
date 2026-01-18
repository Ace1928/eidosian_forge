from collections import defaultdict
import networkx as nx
def make_H_from_C_N(C, N):
    return {(c1, c2): sum((1 for node in C[c1] if N[node, c2] == 0)) for c1 in C for c2 in C}