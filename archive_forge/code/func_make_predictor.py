import numbers
from heapq import heappop, heappush
from timeit import default_timer as time
import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from ._bitset import set_raw_bitset_from_binned_bitset
from .common import (
from .histogram import HistogramBuilder
from .predictor import TreePredictor
from .splitting import Splitter
from .utils import sum_parallel
def make_predictor(self, binning_thresholds):
    """Make a TreePredictor object out of the current tree.

        Parameters
        ----------
        binning_thresholds : array-like of floats
            Corresponds to the bin_thresholds_ attribute of the BinMapper.
            For each feature, this stores:

            - the bin frontiers for continuous features
            - the unique raw category values for categorical features

        Returns
        -------
        A TreePredictor object.
        """
    predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
    binned_left_cat_bitsets = np.zeros((self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE)
    raw_left_cat_bitsets = np.zeros((self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE)
    _fill_predictor_arrays(predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets, self.root, binning_thresholds, self.n_bins_non_missing)
    return TreePredictor(predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets)