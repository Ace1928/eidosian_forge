import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def manual_unitvec(vec):
    vec = vec.astype(float)
    if sparse.issparse(vec):
        vec_sum_of_squares = vec.multiply(vec)
        unit = 1.0 / np.sqrt(vec_sum_of_squares.sum())
        return vec.multiply(unit)
    elif not sparse.issparse(vec):
        sum_vec_squared = np.sum(vec ** 2)
        vec /= np.sqrt(sum_vec_squared)
        return vec