from __future__ import annotations
from statsmodels.compat.pandas import (
import numbers
import warnings
import numpy as np
from pandas import (
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
def get_prediction_index(start, end, nobs, base_index, index=None, silent=False, index_none=False, index_generated=None, data=None) -> tuple[int, int, int, Index | None]:
    """
    Get the location of a specific key in an index or model row labels

    Parameters
    ----------
    start : label
        The key at which to start prediction. Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels.
    end : label
        The key at which to end prediction (note that this key will be
        *included* in prediction). Depending on the underlying
        model's index, may be an integer, a date (string, datetime object,
        pd.Timestamp, or pd.Period object), or some other object in the
        model's row labels.
    nobs : int
    base_index : pd.Index

    index : pd.Index, optional
        Optionally an index to associate the predicted results to. If None,
        an attempt is made to create an index for the predicted results
        from the model's index or model's row labels.
    silent : bool, optional
        Argument to silence warnings.

    Returns
    -------
    start : int
        The index / observation location at which to begin prediction.
    end : int
        The index / observation location at which to end in-sample
        prediction. The maximum value for this is nobs-1.
    out_of_sample : int
        The number of observations to forecast after the end of the sample.
    prediction_index : pd.Index or None
        The index associated with the prediction results. This index covers
        the range [start, end + out_of_sample]. If the model has no given
        index and no given row labels (i.e. endog/exog is not Pandas), then
        this will be None.

    Notes
    -----
    The arguments `start` and `end` behave differently, depending on if
    they are integer or not. If either is an integer, then it is assumed
    to refer to a *location* in the index, not to an index value. On the
    other hand, if it is a date string or some other type of object, then
    it is assumed to refer to an index *value*. In all cases, the returned
    `start` and `end` values refer to index *locations* (so in the former
    case, the given location is validated and returned whereas in the
    latter case a location is found that corresponds to the given index
    value).

    This difference in behavior is necessary to support `RangeIndex`. This
    is because integers for a RangeIndex could refer either to index values
    or to index locations in an ambiguous way (while for `NumericIndex`,
    since we have required them to be full indexes, there is no ambiguity).
    """
    try:
        start, _, start_oos = get_index_label_loc(start, base_index, data.row_labels)
    except KeyError:
        raise KeyError('The `start` argument could not be matched to a location related to the index of the data.')
    if end is None:
        end = max(start, len(base_index) - 1)
    try:
        end, end_index, end_oos = get_index_label_loc(end, base_index, data.row_labels)
    except KeyError:
        raise KeyError('The `end` argument could not be matched to a location related to the index of the data.')
    if isinstance(start, slice):
        start = start.start
    if isinstance(end, slice):
        end = end.stop - 1
    prediction_index = end_index[start:]
    if end < start:
        raise ValueError('Prediction must have `end` after `start`.')
    if index is not None:
        if not len(prediction_index) == len(index):
            raise ValueError('Invalid `index` provided in prediction. Must have length consistent with `start` and `end` arguments.')
        if not isinstance(data, PandasData) and (not silent):
            warnings.warn('Because the model data (`endog`, `exog`) were not given as Pandas objects, the prediction output will be Numpy arrays, and the given `index` argument will only be used internally.', ValueWarning, stacklevel=2)
        prediction_index = Index(index)
    elif index_generated and (not index_none):
        if data.row_labels is not None and (not (start_oos or end_oos)):
            prediction_index = data.row_labels[start:end + 1]
        else:
            if not silent:
                warnings.warn('No supported index is available. Prediction results will be given with an integer index beginning at `start`.', ValueWarning, stacklevel=2)
            warnings.warn('No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.', FutureWarning, stacklevel=2)
    elif index_none:
        prediction_index = None
    if prediction_index is not None:
        data.predict_start = prediction_index[0]
        data.predict_end = prediction_index[-1]
        data.predict_dates = prediction_index
    else:
        data.predict_start = None
        data.predict_end = None
        data.predict_dates = None
    out_of_sample = max(end - (nobs - 1), 0)
    end -= out_of_sample
    return (start, end, out_of_sample, prediction_index)