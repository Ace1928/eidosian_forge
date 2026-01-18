import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.random import sample_without_replacement
@validate_params({'n_samples': [Interval(Integral, 1, None, closed='left')], 'noise': [Interval(Real, 0, None, closed='left')], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def make_friedman3(n_samples=100, *, noise=0.0, random_state=None):
    """Generate the "Friedman #3" regression problem.

    This dataset is described in Friedman [1] and Breiman [2].

    Inputs `X` are 4 independent features uniformly distributed on the
    intervals::

        0 <= X[:, 0] <= 100,
        40 * pi <= X[:, 1] <= 560 * pi,
        0 <= X[:, 2] <= 1,
        1 <= X[:, 3] <= 11.

    The output `y` is created according to the formula::

        y(X) = arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) + noise * N(0, 1).

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, 4)
        The input samples.

    y : ndarray of shape (n_samples,)
        The output values.

    References
    ----------
    .. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
           of Statistics 19 (1), pages 1-67, 1991.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
           pages 123-140, 1996.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman3
    >>> X, y = make_friedman3(random_state=42)
    >>> X.shape
    (100, 4)
    >>> y.shape
    (100,)
    >>> list(y[:3])
    [1.5..., 0.9..., 0.4...]
    """
    generator = check_random_state(random_state)
    X = generator.uniform(size=(n_samples, 4))
    X[:, 0] *= 100
    X[:, 1] *= 520 * np.pi
    X[:, 1] += 40 * np.pi
    X[:, 3] *= 10
    X[:, 3] += 1
    y = np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) + noise * generator.standard_normal(size=n_samples)
    return (X, y)