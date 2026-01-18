import warnings
from abc import ABCMeta, abstractmethod
from operator import attrgetter
import numpy as np
from scipy.sparse import csc_matrix, issparse
from ..base import TransformerMixin
from ..utils import (
from ..utils._set_output import _get_output_config
from ..utils._tags import _safe_tags
from ..utils.validation import _check_feature_names_in, check_is_fitted
def get_support(self, indices=False):
    """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
    mask = self._get_support_mask()
    return mask if not indices else np.where(mask)[0]