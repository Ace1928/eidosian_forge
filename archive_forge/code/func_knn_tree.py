from __future__ import division
from . import matrix
from . import utils
from .base import DataGraph
from .base import PyGSPGraph
from builtins import super
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import numbers
import numpy as np
import tasklogger
import warnings
@property
def knn_tree(self):
    """KNN tree object (cached)

        Builds or returns the fitted KNN tree.
        TODO: can we be more clever than sklearn when it comes to choosing
        between KD tree, ball tree and brute force?

        Returns
        -------
        knn_tree : `sklearn.neighbors.NearestNeighbors`
        """
    try:
        return self._knn_tree
    except AttributeError:
        try:
            self._knn_tree = NearestNeighbors(n_neighbors=self.knn + 1, algorithm='ball_tree', metric=self.distance, n_jobs=self.n_jobs).fit(self.data_nu)
        except ValueError:
            warnings.warn('Metric {} not valid for `sklearn.neighbors.BallTree`. Graph instantiation may be slower than normal.'.format(self.distance), UserWarning)
            self._knn_tree = NearestNeighbors(n_neighbors=self.knn + 1, algorithm='auto', metric=self.distance, n_jobs=self.n_jobs).fit(self.data_nu)
        return self._knn_tree