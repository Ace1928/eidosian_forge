from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
def cell_full(t1_index, t2_index, similarity):
    if dominant and column_sum[t1_index] + abs(similarity) >= 1.0:
        return True
    assert column_nonzero[t1_index] <= nonzero_limit
    if column_nonzero[t1_index] == nonzero_limit:
        return True
    if symmetric and (t1_index, t2_index) in assigned_cells:
        return True
    return False