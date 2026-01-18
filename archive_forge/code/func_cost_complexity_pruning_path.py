import copy
import numbers
from abc import ABCMeta, abstractmethod
from math import ceil
from numbers import Integral, Real
import numpy as np
from scipy.sparse import issparse
from ..base import (
from ..utils import Bunch, check_random_state, compute_sample_weight
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import (
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import (
from ._utils import _any_isnan_axis0
def cost_complexity_pruning_path(self, X, y, sample_weight=None):
    """Compute the pruning path during Minimal Cost-Complexity Pruning.

        See :ref:`minimal_cost_complexity_pruning` for details on the pruning
        process.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        Returns
        -------
        ccp_path : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            ccp_alphas : ndarray
                Effective alphas of subtree during pruning.

            impurities : ndarray
                Sum of the impurities of the subtree leaves for the
                corresponding alpha value in ``ccp_alphas``.
        """
    est = clone(self).set_params(ccp_alpha=0.0)
    est.fit(X, y, sample_weight=sample_weight)
    return Bunch(**ccp_pruning_path(est.tree_))