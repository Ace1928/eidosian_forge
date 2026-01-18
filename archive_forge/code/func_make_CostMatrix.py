import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def make_CostMatrix(C, m, n):
    lsa_row_ind, lsa_col_ind = sp.optimize.linear_sum_assignment(C)
    indexes = zip(range(len(lsa_row_ind)), lsa_row_ind, lsa_col_ind)
    subst_ind = [k for k, i, j in indexes if i < m and j < n]
    indexes = zip(range(len(lsa_row_ind)), lsa_row_ind, lsa_col_ind)
    dummy_ind = [k for k, i, j in indexes if i >= m and j >= n]
    lsa_row_ind[dummy_ind] = lsa_col_ind[subst_ind] + m
    lsa_col_ind[dummy_ind] = lsa_row_ind[subst_ind] + n
    return CostMatrix(C, lsa_row_ind, lsa_col_ind, C[lsa_row_ind, lsa_col_ind].sum())