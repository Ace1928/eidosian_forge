import warnings
from numbers import Integral, Real
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox
from ..base import (
from ..utils import _array_api, check_array
from ..utils._array_api import get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs import (
from ..utils.sparsefuncs_fast import (
from ..utils.validation import (
from ._encoders import OneHotEncoder
@validate_params({'X': ['array-like'], 'axis': [Options(Integral, {0, 1})]}, prefer_skip_nested_validation=False)
def minmax_scale(X, feature_range=(0, 1), *, axis=0, copy=True):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by (when ``axis=0``)::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    The transformation is calculated as (when ``axis=0``)::

       X_scaled = scale * X + min - X.min(axis=0) * scale
       where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    .. versionadded:: 0.17
       *minmax_scale* function interface
       to :class:`~sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    axis : {0, 1}, default=0
        Axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : ndarray of shape (n_samples, n_features)
        The transformed data.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.minmax_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.MinMaxScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(MinMaxScaler(), LogisticRegression())`.

    See Also
    --------
    MinMaxScaler : Performs scaling to a given range using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    Examples
    --------
    >>> from sklearn.preprocessing import minmax_scale
    >>> X = [[-2, 1, 2], [-1, 0, 1]]
    >>> minmax_scale(X, axis=0)  # scale each column independently
    array([[0., 1., 1.],
           [1., 0., 0.]])
    >>> minmax_scale(X, axis=1)  # scale each row independently
    array([[0.  , 0.75, 1.  ],
           [0.  , 0.5 , 1.  ]])
    """
    X = check_array(X, copy=False, ensure_2d=False, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    original_ndim = X.ndim
    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)
    s = MinMaxScaler(feature_range=feature_range, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T
    if original_ndim == 1:
        X = X.ravel()
    return X