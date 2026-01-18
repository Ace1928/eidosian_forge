import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def diff_endog(self, new_endog, tolerance=1e-10):
    endog = self.endog.T
    if len(new_endog) < len(endog):
        raise ValueError('Given data (length %d) is too short to diff against model data (length %d).' % (len(new_endog), len(endog)))
    if len(new_endog) > len(endog):
        nobs_append = len(new_endog) - len(endog)
        endog = np.c_[endog.T, new_endog[-nobs_append:].T * np.nan].T
    new_nan = np.isnan(new_endog)
    existing_nan = np.isnan(endog)
    diff = np.abs(new_endog - endog)
    diff[new_nan ^ existing_nan] = np.inf
    diff[new_nan & existing_nan] = 0.0
    is_revision = diff > tolerance
    is_new = existing_nan & ~new_nan
    is_revision[is_new] = False
    revision_ix = list(zip(*np.where(is_revision)))
    new_ix = list(zip(*np.where(is_new)))
    return (revision_ix, new_ix)