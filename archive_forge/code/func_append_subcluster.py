import warnings
from math import sqrt
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances_argmin
from ..metrics.pairwise import euclidean_distances
from ..utils._param_validation import Interval
from ..utils.extmath import row_norms
from ..utils.validation import check_is_fitted
from . import AgglomerativeClustering
def append_subcluster(self, subcluster):
    n_samples = len(self.subclusters_)
    self.subclusters_.append(subcluster)
    self.init_centroids_[n_samples] = subcluster.centroid_
    self.init_sq_norm_[n_samples] = subcluster.sq_norm_
    self.centroids_ = self.init_centroids_[:n_samples + 1, :]
    self.squared_norm_ = self.init_sq_norm_[:n_samples + 1]