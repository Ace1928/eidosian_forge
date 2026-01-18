import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
def one_pass_var(X):
    n = X.shape[0]
    exp_x2 = (X ** 2).sum(axis=0) / n
    expx_2 = (X.sum(axis=0) / n) ** 2
    return exp_x2 - expx_2