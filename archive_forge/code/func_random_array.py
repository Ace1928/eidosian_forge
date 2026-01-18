import numbers
import math
import numpy as np
from scipy._lib._util import check_random_state, rng_integers
from ._sputils import upcast, get_index_dtype, isscalarlike
from ._sparsetools import csr_hstack
from ._bsr import bsr_matrix, bsr_array
from ._coo import coo_matrix, coo_array
from ._csc import csc_matrix, csc_array
from ._csr import csr_matrix, csr_array
from ._dia import dia_matrix, dia_array
from ._base import issparse, sparray
def random_array(shape, *, density=0.01, format='coo', dtype=None, random_state=None, data_sampler=None):
    """Return a sparse array of uniformly random numbers in [0, 1)

    Returns a sparse array with the given shape and density
    where values are generated uniformly randomly in the range [0, 1).

    .. warning::

        Since numpy 1.17, passing a ``np.random.Generator`` (e.g.
        ``np.random.default_rng``) for ``random_state`` will lead to much
        faster execution times.

        A much slower implementation is used by default for backwards
        compatibility.

    Parameters
    ----------
    shape : int or tuple of ints
        shape of the array
    density : real, optional (default: 0.01)
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional (default: 'coo')
        sparse matrix format.
    dtype : dtype, optional (default: np.float64)
        type of the returned matrix values.
    random_state : {None, int, `Generator`, `RandomState`}, optional
        A random number generator to determine nonzero structure. We recommend using
        a `numpy.random.Generator` manually provided for every call as it is much
        faster than RandomState.

        - If `None` (or `np.random`), the `numpy.random.RandomState`
          singleton is used.
        - If an int, a new ``Generator`` instance is used,
          seeded with the int.
        - If a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        This random state will be used for sampling `indices` (the sparsity
        structure), and by default for the data values too (see `data_sampler`).

    data_sampler : callable, optional (default depends on dtype)
        Sampler of random data values with keyword arg `size`.
        This function should take a single keyword argument `size` specifying
        the length of its returned ndarray. It is used to generate the nonzero
        values in the matrix after the locations of those values are chosen.
        By default, uniform [0, 1) random values are used unless `dtype` is
        an integer (default uniform integers from that dtype) or
        complex (default uniform over the unit square in the complex plane).
        For these, the `random_state` rng is used e.g. `rng.uniform(size=size)`.

    Returns
    -------
    res : sparse array

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import numpy as np
    >>> import scipy as sp
    >>> rng = np.random.default_rng()

    Default sampling uniformly from [0, 1):

    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng)

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            random_state=rng, data_sampler=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, random_state=rng):
    ...     return random_state.standard_normal(size) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng,
    ...                      data_sampler=np_normal_squared)

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, random_state=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=random_state) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng,
    ...                      data_sampler=sp_stats_normal_squared)

    Or we can subclass sp.stats rv_continous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return random_state.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X().rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            random_state=rng, data_sampler=Y)
    """
    if random_state is None:
        random_state = np.random.default_rng()
    data, ind = _random(shape, density, format, dtype, random_state, data_sampler)
    return coo_array((data, ind), shape=shape).asformat(format)