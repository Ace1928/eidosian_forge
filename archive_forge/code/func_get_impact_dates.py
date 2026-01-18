import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def get_impact_dates(previous_model, updated_model, impact_date=None, start=None, end=None, periods=None):
    """
    Compute start/end periods and an index, often for impacts of data updates

    Parameters
    ----------
    previous_model : MLEModel
        Model used to compute default start/end periods if None are given.
        In the case of computing impacts of data updates, this would be the
        model estimated with the previous dataset. Otherwise, can be the same
        as `updated_model`.
    updated_model : MLEModel
        Model used to compute the index. In the case of computing impacts of
        data updates, this would be the model estimated with the updated
        dataset. Otherwise, can be the same as `previous_model`.
    impact_date : {int, str, datetime}, optional
        Specific individual impact date. Cannot be used in combination with
        `start`, `end`, or `periods`.
    start : {int, str, datetime}, optional
        Starting point of the impact dates. If given, one of `end` or `periods`
        must also be given. If a negative integer, will be computed relative to
        the dates in the `updated_model` index. Cannot be used in combination
        with `impact_date`.
    end : {int, str, datetime}, optional
        Ending point of the impact dates. If given, one of `start` or `periods`
        must also be given. If a negative integer, will be computed relative to
        the dates in the `updated_model` index. Cannot be used in combination
        with `impact_date`.
    periods : int, optional
        Number of impact date periods. If given, one of `start` or `end`
        must also be given. Cannot be used in combination with `impact_date`.

    Returns
    -------
    start : int
        Integer location of the first included impact dates.
    end : int
        Integer location of the last included impact dates (i.e. this integer
        location is included in the returned `index`).
    index : pd.Index
        Index associated with `start` and `end`, as computed from the
        `updated_model`'s index.

    Notes
    -----
    This function is typically used as a helper for standardizing start and
    end periods for a date range where the most sensible default values are
    based on some initial dataset (here contained in the `previous_model`),
    while index-related operations (especially relative start/end dates given
    via negative integers) are most sensibly computed from an updated dataset
    (here contained in the `updated_model`).

    """
    if impact_date is not None:
        if not (start is None and end is None and (periods is None)):
            raise ValueError('Cannot use the `impact_date` argument in combination with `start`, `end`, or `periods`.')
        start = impact_date
        periods = 1
    if start is None and end is None and (periods is None):
        start = previous_model.nobs - 1
        end = previous_model.nobs - 1
    if int(start is None) + int(end is None) + int(periods is None) != 1:
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    elif start is not None and periods is not None:
        start, _, _, _ = updated_model._get_prediction_index(start, start)
        end = start + (periods - 1)
    elif end is not None and periods is not None:
        _, end, _, _ = updated_model._get_prediction_index(end, end)
        start = end - (periods - 1)
    elif start is not None and end is not None:
        pass
    start, end, out_of_sample, prediction_index = updated_model._get_prediction_index(start, end)
    end = end + out_of_sample
    return (start, end, prediction_index)