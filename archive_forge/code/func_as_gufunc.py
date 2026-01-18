from __future__ import annotations
import re
import numpy as np
from tlz import concat, merge, unique
from dask.array.core import Array, apply_infer_dtype, asarray, blockwise, getitem
from dask.array.utils import meta_from_array
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
def as_gufunc(signature=None, **kwargs):
    """
    Decorator for ``dask.array.gufunc``.

    Parameters
    ----------
    signature : String
        Specifies what core dimensions are consumed and produced by ``func``.
        According to the specification of numpy.gufunc signature [2]_
    axes: List of tuples, optional, keyword only
        A list of tuples with indices of axes a generalized ufunc should operate on.
        For instance, for a signature of ``"(i,j),(j,k)->(i,k)"`` appropriate for
        matrix multiplication, the base elements are two-dimensional matrices
        and these are taken to be stored in the two last axes of each argument. The
        corresponding axes keyword would be ``[(-2, -1), (-2, -1), (-2, -1)]``.
        For simplicity, for generalized ufuncs that operate on 1-dimensional arrays
        (vectors), a single integer is accepted instead of a single-element tuple,
        and for generalized ufuncs for which all outputs are scalars, the output
        tuples can be omitted.
    axis: int, optional, keyword only
        A single axis over which a generalized ufunc should operate. This is a short-cut
        for ufuncs that operate over a single, shared core dimension, equivalent to passing
        in axes with entries of (axis,) for each single-core-dimension argument and ``()`` for
        all others. For instance, for a signature ``"(i),(i)->()"``, it is equivalent to passing
        in ``axes=[(axis,), (axis,), ()]``.
    keepdims: bool, optional, keyword only
        If this is set to True, axes which are reduced over will be left in the result as
        a dimension with size one, so that the result will broadcast correctly against the
        inputs. This option can only be used for generalized ufuncs that operate on inputs
        that all have the same number of core dimensions and with outputs that have no core
        dimensions , i.e., with signatures like ``"(i),(i)->()"`` or ``"(m,m)->()"``.
        If used, the location of the dimensions in the output can be controlled with axes
        and axis.
    output_dtypes : Optional, dtype or list of dtypes, keyword only
        Valid numpy dtype specification or list thereof.
        If not given, a call of ``func`` with a small set of data
        is performed in order to try to automatically determine the
        output dtypes.
    output_sizes : dict, optional, keyword only
        Optional mapping from dimension names to sizes for outputs. Only used if
        new core dimensions (not found on inputs) appear on outputs.
    vectorize: bool, keyword only
        If set to ``True``, ``np.vectorize`` is applied to ``func`` for
        convenience. Defaults to ``False``.
    allow_rechunk: Optional, bool, keyword only
        Allows rechunking, otherwise chunk sizes need to match and core
        dimensions are to consist only of one chunk.
        Warning: enabling this can increase memory usage significantly.
        Defaults to ``False``.
    meta: Optional, tuple, keyword only
        tuple of empty ndarrays describing the shape and dtype of the output of the gufunc.
        Defaults to ``None``.

    Returns
    -------
    Decorator for `pyfunc` that itself returns a `gufunc`.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> a = da.random.normal(size=(10,20,30), chunks=(5, 10, 30))
    >>> @da.as_gufunc("(i)->(),()", output_dtypes=(float, float))
    ... def stats(x):
    ...     return np.mean(x, axis=-1), np.std(x, axis=-1)
    >>> mean, std = stats(a)
    >>> mean.compute().shape
    (10, 20)

    >>> a = da.random.normal(size=(   20,30), chunks=(10, 30))
    >>> b = da.random.normal(size=(10, 1,40), chunks=(5, 1, 40))
    >>> @da.as_gufunc("(i),(j)->(i,j)", output_dtypes=float, vectorize=True)
    ... def outer_product(x, y):
    ...     return np.einsum("i,j->ij", x, y)
    >>> c = outer_product(a, b)
    >>> c.compute().shape
    (10, 20, 30, 40)

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/reference/ufuncs.html
    .. [2] https://docs.scipy.org/doc/numpy/reference/c-api/generalized-ufuncs.html
    """
    _allowedkeys = {'vectorize', 'axes', 'axis', 'keepdims', 'output_sizes', 'output_dtypes', 'allow_rechunk', 'meta'}
    if kwargs.keys() - _allowedkeys:
        raise TypeError('Unsupported keyword argument(s) provided')

    def _as_gufunc(pyfunc):
        return gufunc(pyfunc, signature=signature, **kwargs)
    _as_gufunc.__doc__ = "\n        Decorator to make ``dask.array.gufunc``\n        signature: ``'{signature}'``\n\n        Parameters\n        ----------\n        pyfunc : callable\n            Function matching signature ``'{signature}'``.\n\n        Returns\n        -------\n        ``dask.array.gufunc``\n        ".format(signature=signature)
    return _as_gufunc