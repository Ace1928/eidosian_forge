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
def insert_cf_subcluster(self, subcluster):
    """Insert a new subcluster into the node."""
    if not self.subclusters_:
        self.append_subcluster(subcluster)
        return False
    threshold = self.threshold
    branching_factor = self.branching_factor
    dist_matrix = np.dot(self.centroids_, subcluster.centroid_)
    dist_matrix *= -2.0
    dist_matrix += self.squared_norm_
    closest_index = np.argmin(dist_matrix)
    closest_subcluster = self.subclusters_[closest_index]
    if closest_subcluster.child_ is not None:
        split_child = closest_subcluster.child_.insert_cf_subcluster(subcluster)
        if not split_child:
            closest_subcluster.update(subcluster)
            self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
            self.init_sq_norm_[closest_index] = self.subclusters_[closest_index].sq_norm_
            return False
        else:
            new_subcluster1, new_subcluster2 = _split_node(closest_subcluster.child_, threshold, branching_factor)
            self.update_split_subclusters(closest_subcluster, new_subcluster1, new_subcluster2)
            if len(self.subclusters_) > self.branching_factor:
                return True
            return False
    else:
        merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
        if merged:
            self.init_centroids_[closest_index] = closest_subcluster.centroid_
            self.init_sq_norm_[closest_index] = closest_subcluster.sq_norm_
            return False
        elif len(self.subclusters_) < self.branching_factor:
            self.append_subcluster(subcluster)
            return False
        else:
            self.append_subcluster(subcluster)
            return True