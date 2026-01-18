import itertools
import networkx as nx
def cc_max(nu, nv):
    return len(nu & nv) / max(len(nu), len(nv))