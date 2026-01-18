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
def extend_to_data(self, Y):
    """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of samples in `self.data`. Any
        transformation of `self.data` can be trivially applied to `Y` by
        performing

        `transform_Y = self.interpolate(transform, transitions)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        transitions : array-like, shape=[n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
    Y = self._check_extension_shape(Y)
    kernel = self.build_kernel_to_data(Y)
    transitions = normalize(kernel, norm='l1', axis=1)
    return transitions