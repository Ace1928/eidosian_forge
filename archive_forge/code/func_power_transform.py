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
@validate_params({'X': ['array-like']}, prefer_skip_nested_validation=False)
def power_transform(X, method='yeo-johnson', *, standardize=True, copy=True):
    """Parametric, monotonic transformation to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, power_transform supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to be transformed using a power transformation.

    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

        .. versionchanged:: 0.23
            The default value of the `method` parameter changed from
            'box-cox' to 'yeo-johnson' in 0.23.

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : bool, default=True
        If False, try to avoid a copy and transform in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_trans : ndarray of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    PowerTransformer : Equivalent transformation with the
        Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    quantile_transform : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import power_transform
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(power_transform(data, method='box-cox'))
    [[-1.332... -0.707...]
     [ 0.256... -0.707...]
     [ 1.076...  1.414...]]

    .. warning:: Risk of data leak.
        Do not use :func:`~sklearn.preprocessing.power_transform` unless you
        know what you are doing. A common mistake is to apply it to the entire
        data *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.PowerTransformer` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking, e.g.: `pipe = make_pipeline(PowerTransformer(),
        LogisticRegression())`.
    """
    pt = PowerTransformer(method=method, standardize=standardize, copy=copy)
    return pt.fit_transform(X)