import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def reduce_C(C, i, j, m, n):
    row_ind = [k not in i and k - m not in j for k in range(m + n)]
    col_ind = [k not in j and k - n not in i for k in range(m + n)]
    return C[row_ind, :][:, col_ind]