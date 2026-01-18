import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def reduce_Ce(Ce, ij, m, n):
    if len(ij):
        i, j = zip(*ij)
        m_i = m - sum((1 for t in i if t < m))
        n_j = n - sum((1 for t in j if t < n))
        return make_CostMatrix(reduce_C(Ce.C, i, j, m, n), m_i, n_j)
    return Ce