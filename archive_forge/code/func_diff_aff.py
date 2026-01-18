from . import matrix
from . import utils
from builtins import super
from copy import copy as shallow_copy
from future.utils import with_metaclass
from inspect import signature
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import abc
import numbers
import numpy as np
import pickle
import pygsp
import sys
import tasklogger
import warnings
@property
def diff_aff(self):
    """Symmetric diffusion affinity matrix

        Return or calculate the symmetric diffusion affinity matrix

        .. math:: A(x,y) = K(x,y) (d(x) d(y))^{-1/2}

        where :math:`d` is the degrees (row sums of the kernel.)

        Returns
        -------

        diff_aff : array-like, shape=[n_samples, n_samples]
            symmetric diffusion affinity matrix defined as a
            doubly-stochastic form of the kernel matrix
        """
    row_degrees = self.kernel_degree
    if sparse.issparse(self.kernel):
        degrees = sparse.csr_matrix((1 / np.sqrt(row_degrees.flatten()), np.arange(len(row_degrees)), np.arange(len(row_degrees) + 1)))
        return degrees @ self.kernel @ degrees
    else:
        col_degrees = row_degrees.T
        return self.kernel / np.sqrt(row_degrees) / np.sqrt(col_degrees)