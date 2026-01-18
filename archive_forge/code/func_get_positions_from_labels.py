import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
def get_positions_from_labels(self, row_loc, col_loc):
    """
        Compute index and column positions from their respective locators.

        Inputs to this method are arguments the the pandas user could pass to loc.
        This function will compute the corresponding index and column positions
        that the user could equivalently pass to iloc.

        Parameters
        ----------
        row_loc : scalar, slice, list, array or tuple
            Row locator.
        col_loc : scalar, slice, list, array or tuple
            Columns locator.

        Returns
        -------
        row_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of index labels.
        col_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of columns labels.

        Notes
        -----
        Usage of `slice(None)` as a resulting lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
    from modin.pandas.indexing import boolean_mask_to_numeric, is_boolean_array, is_list_like, is_range_like
    lookups = []
    for axis, axis_loc in enumerate((row_loc, col_loc)):
        if is_scalar(axis_loc):
            axis_loc = np.array([axis_loc])
        if isinstance(axis_loc, pandas.RangeIndex):
            axis_lookup = axis_loc
        elif isinstance(axis_loc, slice) or is_range_like(axis_loc):
            if isinstance(axis_loc, slice) and axis_loc == slice(None):
                axis_lookup = axis_loc
            else:
                axis_labels = self.get_axis(axis)
                if axis_loc.stop is None or not is_number(axis_loc.stop):
                    slice_stop = axis_loc.stop
                else:
                    slice_stop = axis_loc.stop - (0 if axis_loc.step is None else axis_loc.step)
                axis_lookup = axis_labels.slice_indexer(axis_loc.start, slice_stop, axis_loc.step)
                axis_lookup = pandas.RangeIndex(start=axis_lookup.start if axis_lookup.start >= 0 else axis_lookup.start + len(axis_labels), stop=axis_lookup.stop if axis_lookup.stop >= 0 else axis_lookup.stop + len(axis_labels), step=axis_lookup.step)
        elif self.has_multiindex(axis):
            if isinstance(axis_loc, pandas.MultiIndex):
                axis_lookup = self.get_axis(axis).get_indexer_for(axis_loc)
            else:
                axis_lookup = self.get_axis(axis).get_locs(axis_loc)
        elif is_boolean_array(axis_loc):
            axis_lookup = boolean_mask_to_numeric(axis_loc)
        else:
            axis_labels = self.get_axis(axis)
            if is_list_like(axis_loc) and (not isinstance(axis_loc, (np.ndarray, pandas.Index))):
                axis_loc = np.array(axis_loc, dtype=axis_labels.dtype)
            axis_lookup = axis_labels.get_indexer_for(axis_loc)
            missing_mask = axis_lookup == -1
            if missing_mask.any():
                missing_labels = axis_loc[missing_mask] if is_list_like(axis_loc) else axis_loc
                raise KeyError(missing_labels)
        if isinstance(axis_lookup, pandas.Index) and (not is_range_like(axis_lookup)):
            axis_lookup = axis_lookup.values
        lookups.append(axis_lookup)
    return lookups