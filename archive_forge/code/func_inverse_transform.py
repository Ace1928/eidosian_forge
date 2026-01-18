import array
import itertools
import warnings
from collections import defaultdict
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import column_or_1d
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, validate_params
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import min_max_axis
from ..utils.validation import _num_samples, check_array, check_is_fitted
def inverse_transform(self, yt):
    """Transform the given indicator matrix into label sets.

        Parameters
        ----------
        yt : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            A matrix containing only 1s ands 0s.

        Returns
        -------
        y : list of tuples
            The set of labels for each sample such that `y[i]` consists of
            `classes_[j]` for each `yt[i, j] == 1`.
        """
    check_is_fitted(self)
    if yt.shape[1] != len(self.classes_):
        raise ValueError('Expected indicator for {0} classes, but got {1}'.format(len(self.classes_), yt.shape[1]))
    if sp.issparse(yt):
        yt = yt.tocsr()
        if len(yt.data) != 0 and len(np.setdiff1d(yt.data, [0, 1])) > 0:
            raise ValueError('Expected only 0s and 1s in label indicator.')
        return [tuple(self.classes_.take(yt.indices[start:end])) for start, end in zip(yt.indptr[:-1], yt.indptr[1:])]
    else:
        unexpected = np.setdiff1d(yt, [0, 1])
        if len(unexpected) > 0:
            raise ValueError('Expected only 0s and 1s in label indicator. Also got {0}'.format(unexpected))
        return [tuple(self.classes_.compress(indicators)) for indicators in yt]