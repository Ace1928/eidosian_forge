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
@validate_params({'n_samples': [Interval(Integral, 1, None, closed='left')], 'n_components': [Interval(Integral, 1, None, closed='left')], 'n_features': [Interval(Integral, 1, None, closed='left')], 'n_nonzero_coefs': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state'], 'data_transposed': ['boolean', Hidden(StrOptions({'deprecated'}))]}, prefer_skip_nested_validation=True)
def make_sparse_coded_signal(n_samples, *, n_components, n_features, n_nonzero_coefs, random_state=None, data_transposed='deprecated'):
    """Generate a signal as a sparse combination of dictionary elements.

    Returns a matrix `Y = DX`, such that `D` is of shape `(n_features, n_components)`,
    `X` is of shape `(n_components, n_samples)` and each column of `X` has exactly
    `n_nonzero_coefs` non-zero elements.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    n_components : int
        Number of components in the dictionary.

    n_features : int
        Number of features of the dataset to generate.

    n_nonzero_coefs : int
        Number of active (non-zero) coefficients in each sample.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    data_transposed : bool, default=False
        By default, Y, D and X are not transposed.

        .. versionadded:: 1.1

        .. versionchanged:: 1.3
            Default value changed from True to False.

        .. deprecated:: 1.3
            `data_transposed` is deprecated and will be removed in 1.5.

    Returns
    -------
    data : ndarray of shape (n_features, n_samples) or (n_samples, n_features)
        The encoded signal (Y). The shape is `(n_samples, n_features)` if
        `data_transposed` is False, otherwise it's `(n_features, n_samples)`.

    dictionary : ndarray of shape (n_features, n_components) or             (n_components, n_features)
        The dictionary with normalized components (D). The shape is
        `(n_components, n_features)` if `data_transposed` is False, otherwise it's
        `(n_features, n_components)`.

    code : ndarray of shape (n_components, n_samples) or (n_samples, n_components)
        The sparse code such that each column of this matrix has exactly
        n_nonzero_coefs non-zero items (X). The shape is `(n_samples, n_components)`
        if `data_transposed` is False, otherwise it's `(n_components, n_samples)`.
    """
    generator = check_random_state(random_state)
    D = generator.standard_normal(size=(n_features, n_components))
    D /= np.sqrt(np.sum(D ** 2, axis=0))
    X = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        idx = np.arange(n_components)
        generator.shuffle(idx)
        idx = idx[:n_nonzero_coefs]
        X[idx, i] = generator.standard_normal(size=n_nonzero_coefs)
    Y = np.dot(D, X)
    if data_transposed != 'deprecated':
        warnings.warn('data_transposed was deprecated in version 1.3 and will be removed in 1.5.', FutureWarning)
    else:
        data_transposed = False
    if not data_transposed:
        Y, D, X = (Y.T, D.T, X.T)
    return map(np.squeeze, (Y, D, X))