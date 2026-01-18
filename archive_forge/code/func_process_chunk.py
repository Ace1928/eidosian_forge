from numbers import Integral
import numpy as np
from ..base import _fit_context
from ..metrics import pairwise_distances_chunked
from ..metrics.pairwise import _NAN_METRICS
from ..neighbors._base import _get_weights
from ..utils import is_scalar_nan
from ..utils._mask import _get_mask
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
from ._base import _BaseImputer
def process_chunk(dist_chunk, start):
    row_missing_chunk = row_missing_idx[start:start + len(dist_chunk)]
    for col in range(X.shape[1]):
        if not valid_mask[col]:
            continue
        col_mask = mask[row_missing_chunk, col]
        if not np.any(col_mask):
            continue
        potential_donors_idx, = np.nonzero(non_missing_fix_X[:, col])
        receivers_idx = row_missing_chunk[np.flatnonzero(col_mask)]
        dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][:, potential_donors_idx]
        all_nan_dist_mask = np.isnan(dist_subset).all(axis=1)
        all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]
        if all_nan_receivers_idx.size:
            col_mean = np.ma.array(self._fit_X[:, col], mask=mask_fit_X[:, col]).mean()
            X[all_nan_receivers_idx, col] = col_mean
            if len(all_nan_receivers_idx) == len(receivers_idx):
                continue
            receivers_idx = receivers_idx[~all_nan_dist_mask]
            dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][:, potential_donors_idx]
        n_neighbors = min(self.n_neighbors, len(potential_donors_idx))
        value = self._calc_impute(dist_subset, n_neighbors, self._fit_X[potential_donors_idx, col], mask_fit_X[potential_donors_idx, col])
        X[receivers_idx, col] = value