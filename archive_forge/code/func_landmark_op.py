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
def landmark_op(self):
    """Landmark operator

        Compute or return the landmark operator

        Returns
        -------
        landmark_op : array-like, shape=[n_landmark, n_landmark]
            Landmark operator. Can be treated as a diffusion operator between
            landmarks.
        """
    try:
        return self._landmark_op
    except AttributeError:
        self.build_landmark_op()
        return self._landmark_op