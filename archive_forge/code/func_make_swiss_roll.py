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
@validate_params({'n_samples': [Interval(Integral, 1, None, closed='left')], 'noise': [Interval(Real, 0, None, closed='left')], 'random_state': ['random_state'], 'hole': ['boolean']}, prefer_skip_nested_validation=True)
def make_swiss_roll(n_samples=100, *, noise=0.0, random_state=None, hole=False):
    """Generate a swiss roll dataset.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of sample points on the Swiss Roll.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    hole : bool, default=False
        If True generates the swiss roll with hole dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The points.

    t : ndarray of shape (n_samples,)
        The univariate position of the sample according to the main dimension
        of the points in the manifold.

    Notes
    -----
    The algorithm is from Marsland [1].

    References
    ----------
    .. [1] S. Marsland, "Machine Learning: An Algorithmic Perspective", 2nd edition,
           Chapter 6, 2014.
           https://homepages.ecs.vuw.ac.nz/~marslast/Code/Ch6/lle.py
    """
    generator = check_random_state(random_state)
    if not hole:
        t = 1.5 * np.pi * (1 + 2 * generator.uniform(size=n_samples))
        y = 21 * generator.uniform(size=n_samples)
    else:
        corners = np.array([[np.pi * (1.5 + i), j * 7] for i in range(3) for j in range(3)])
        corners = np.delete(corners, 4, axis=0)
        corner_index = generator.choice(8, n_samples)
        parameters = generator.uniform(size=(2, n_samples)) * np.array([[np.pi], [7]])
        t, y = corners[corner_index].T + parameters
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.vstack((x, y, z))
    X += noise * generator.standard_normal(size=(3, n_samples))
    X = X.T
    t = np.squeeze(t)
    return (X, t)