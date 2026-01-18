from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast, overload
import numpy as np
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.alignment import align, deep_align
from xarray.core.common import zeros_like
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.formatting import limit_lines
from xarray.core.indexes import Index, filter_indexes_from_coords
from xarray.core.merge import merge_attrs, merge_coordinates_without_align
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Dims, T_DataArray
from xarray.core.utils import is_dict_like, is_duck_dask_array, is_scalar, parse_dims
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.util.deprecation_helpers import deprecate_dims
def apply_ufunc(func: Callable, *args: Any, input_core_dims: Sequence[Sequence] | None=None, output_core_dims: Sequence[Sequence] | None=((),), exclude_dims: Set=frozenset(), vectorize: bool=False, join: JoinOptions='exact', dataset_join: str='exact', dataset_fill_value: object=_NO_FILL_VALUE, keep_attrs: bool | str | None=None, kwargs: Mapping | None=None, dask: Literal['forbidden', 'allowed', 'parallelized']='forbidden', output_dtypes: Sequence | None=None, output_sizes: Mapping[Any, int] | None=None, meta: Any=None, dask_gufunc_kwargs: dict[str, Any] | None=None, on_missing_core_dim: MissingCoreDimOptions='raise') -> Any:
    """Apply a vectorized function for unlabeled arrays on xarray objects.

    The function will be mapped over the data variable(s) of the input
    arguments using xarray's standard rules for labeled computation, including
    alignment, broadcasting, looping over GroupBy/Dataset variables, and
    merging of coordinates.

    Parameters
    ----------
    func : callable
        Function to call like ``func(*args, **kwargs)`` on unlabeled arrays
        (``.data``) that returns an array or tuple of arrays. If multiple
        arguments with non-matching dimensions are supplied, this function is
        expected to vectorize (broadcast) over axes of positional arguments in
        the style of NumPy universal functions [1]_ (if this is not the case,
        set ``vectorize=True``). If this function returns multiple outputs, you
        must set ``output_core_dims`` as well.
    *args : Dataset, DataArray, DataArrayGroupBy, DatasetGroupBy, Variable,         numpy.ndarray, dask.array.Array or scalar
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    input_core_dims : sequence of sequence, optional
        List of the same length as ``args`` giving the list of core dimensions
        on each input argument that should not be broadcast. By default, we
        assume there are no core dimensions on any input arguments.

        For example, ``input_core_dims=[[], ['time']]`` indicates that all
        dimensions on the first argument and all dimensions other than 'time'
        on the second argument should be broadcast.

        Core dimensions are automatically moved to the last axes of input
        variables before applying ``func``, which facilitates using NumPy style
        generalized ufuncs [2]_.
    output_core_dims : list of tuple, optional
        List of the same length as the number of output arguments from
        ``func``, giving the list of core dimensions on each output that were
        not broadcast on the inputs. By default, we assume that ``func``
        outputs exactly one array, with axes corresponding to each broadcast
        dimension.

        Core dimensions are assumed to appear as the last dimensions of each
        output in the provided order.
    exclude_dims : set, optional
        Core dimensions on the inputs to exclude from alignment and
        broadcasting entirely. Any input coordinates along these dimensions
        will be dropped. Each excluded dimension must also appear in
        ``input_core_dims`` for at least one argument. Only dimensions listed
        here are allowed to change size between input and output objects.
    vectorize : bool, optional
        If True, then assume ``func`` only takes arrays defined over core
        dimensions as input and vectorize it automatically with
        :py:func:`numpy.vectorize`. This option exists for convenience, but is
        almost always slower than supplying a pre-vectorized function.
    join : {"outer", "inner", "left", "right", "exact"}, default: "exact"
        Method for joining the indexes of the passed objects along each
        dimension, and the variables of Dataset objects with mismatched
        data variables:

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': raise `ValueError` instead of aligning when indexes to be
          aligned are not equal
    dataset_join : {"outer", "inner", "left", "right", "exact"}, default: "exact"
        Method for joining variables of Dataset objects with mismatched
        data variables.

        - 'outer': take variables from both Dataset objects
        - 'inner': take only overlapped variables
        - 'left': take only variables from the first object
        - 'right': take only variables from the last object
        - 'exact': data variables on all Dataset objects must match exactly
    dataset_fill_value : optional
        Value used in place of missing variables on Dataset inputs when the
        datasets do not share the exact same ``data_vars``. Required if
        ``dataset_join not in {'inner', 'exact'}``, otherwise ignored.
    keep_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", "override"} or bool, optional
        - 'drop' or False: empty attrs on returned xarray object.
        - 'identical': all attrs must be the same on every object.
        - 'no_conflicts': attrs from all objects are combined, any that have the same name must also have the same value.
        - 'drop_conflicts': attrs from all objects are combined, any that have the same name but different values are dropped.
        - 'override' or True: skip comparing and copy attrs from the first object to the result.
    kwargs : dict, optional
        Optional keyword arguments passed directly on to call ``func``.
    dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
        How to handle applying to objects containing lazy data in the form of
        dask arrays:

        - 'forbidden' (default): raise an error if a dask array is encountered.
        - 'allowed': pass dask arrays directly on to ``func``. Prefer this option if
          ``func`` natively supports dask arrays.
        - 'parallelized': automatically parallelize ``func`` if any of the
          inputs are a dask array by using :py:func:`dask.array.apply_gufunc`. Multiple output
          arguments are supported. Only use this option if ``func`` does not natively
          support dask arrays (e.g. converts them to numpy arrays).
    dask_gufunc_kwargs : dict, optional
        Optional keyword arguments passed to :py:func:`dask.array.apply_gufunc` if
        dask='parallelized'. Possible keywords are ``output_sizes``, ``allow_rechunk``
        and ``meta``.
    output_dtypes : list of dtype, optional
        Optional list of output dtypes. Only used if ``dask='parallelized'`` or
        ``vectorize=True``.
    output_sizes : dict, optional
        Optional mapping from dimension names to sizes for outputs. Only used
        if dask='parallelized' and new dimensions (not found on inputs) appear
        on outputs. ``output_sizes`` should be given in the ``dask_gufunc_kwargs``
        parameter. It will be removed as direct parameter in a future version.
    meta : optional
        Size-0 object representing the type of array wrapped by dask array. Passed on to
        :py:func:`dask.array.apply_gufunc`. ``meta`` should be given in the
        ``dask_gufunc_kwargs`` parameter . It will be removed as direct parameter
        a future version.
    on_missing_core_dim : {"raise", "copy", "drop"}, default: "raise"
        How to handle missing core dimensions on input variables.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.

    Notes
    -----
    This function is designed for the more common case where ``func`` can work on numpy
    arrays. If ``func`` needs to manipulate a whole xarray object subset to each block
    it is possible to use :py:func:`xarray.map_blocks`.

    Note that due to the overhead :py:func:`xarray.map_blocks` is considerably slower than ``apply_ufunc``.

    Examples
    --------
    Calculate the vector magnitude of two arguments:

    >>> def magnitude(a, b):
    ...     func = lambda x, y: np.sqrt(x**2 + y**2)
    ...     return xr.apply_ufunc(func, a, b)
    ...

    You can now apply ``magnitude()`` to :py:class:`DataArray` and :py:class:`Dataset`
    objects, with automatically preserved dimensions and coordinates, e.g.,

    >>> array = xr.DataArray([1, 2, 3], coords=[("x", [0.1, 0.2, 0.3])])
    >>> magnitude(array, -array)
    <xarray.DataArray (x: 3)> Size: 24B
    array([1.41421356, 2.82842712, 4.24264069])
    Coordinates:
      * x        (x) float64 24B 0.1 0.2 0.3

    Plain scalars, numpy arrays and a mix of these with xarray objects is also
    supported:

    >>> magnitude(3, 4)
    5.0
    >>> magnitude(3, np.array([0, 4]))
    array([3., 5.])
    >>> magnitude(array, 0)
    <xarray.DataArray (x: 3)> Size: 24B
    array([1., 2., 3.])
    Coordinates:
      * x        (x) float64 24B 0.1 0.2 0.3

    Other examples of how you could use ``apply_ufunc`` to write functions to
    (very nearly) replicate existing xarray functionality:

    Compute the mean (``.mean``) over one dimension:

    >>> def mean(obj, dim):
    ...     # note: apply always moves core dimensions to the end
    ...     return apply_ufunc(
    ...         np.mean, obj, input_core_dims=[[dim]], kwargs={"axis": -1}
    ...     )
    ...

    Inner product over a specific dimension (like :py:func:`dot`):

    >>> def _inner(x, y):
    ...     result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
    ...     return result[..., 0, 0]
    ...
    >>> def inner_product(a, b, dim):
    ...     return apply_ufunc(_inner, a, b, input_core_dims=[[dim], [dim]])
    ...

    Stack objects along a new dimension (like :py:func:`concat`):

    >>> def stack(objects, dim, new_coord):
    ...     # note: this version does not stack coordinates
    ...     func = lambda *x: np.stack(x, axis=-1)
    ...     result = apply_ufunc(
    ...         func,
    ...         *objects,
    ...         output_core_dims=[[dim]],
    ...         join="outer",
    ...         dataset_fill_value=np.nan
    ...     )
    ...     result[dim] = new_coord
    ...     return result
    ...

    If your function is not vectorized but can be applied only to core
    dimensions, you can use ``vectorize=True`` to turn into a vectorized
    function. This wraps :py:func:`numpy.vectorize`, so the operation isn't
    terribly fast. Here we'll use it to calculate the distance between
    empirical samples from two probability distributions, using a scipy
    function that needs to be applied to vectors:

    >>> import scipy.stats
    >>> def earth_mover_distance(first_samples, second_samples, dim="ensemble"):
    ...     return apply_ufunc(
    ...         scipy.stats.wasserstein_distance,
    ...         first_samples,
    ...         second_samples,
    ...         input_core_dims=[[dim], [dim]],
    ...         vectorize=True,
    ...     )
    ...

    Most of NumPy's builtin functions already broadcast their inputs
    appropriately for use in ``apply_ufunc``. You may find helper functions such as
    :py:func:`numpy.broadcast_arrays` helpful in writing your function. ``apply_ufunc`` also
    works well with :py:func:`numba.vectorize` and :py:func:`numba.guvectorize`.

    See Also
    --------
    numpy.broadcast_arrays
    numba.vectorize
    numba.guvectorize
    dask.array.apply_gufunc
    xarray.map_blocks

    :ref:`dask.automatic-parallelization`
        User guide describing :py:func:`apply_ufunc` and :py:func:`map_blocks`.

    :doc:`xarray-tutorial:advanced/apply_ufunc/apply_ufunc`
        Advanced Tutorial on applying numpy function using :py:func:`apply_ufunc`

    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/ufuncs.html
    .. [2] https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.groupby import GroupBy
    from xarray.core.variable import Variable
    if input_core_dims is None:
        input_core_dims = ((),) * len(args)
    elif len(input_core_dims) != len(args):
        raise ValueError(f'input_core_dims must be None or a tuple with the length same to the number of arguments. Given {len(input_core_dims)} input_core_dims: {input_core_dims},  but number of args is {len(args)}.')
    if kwargs is None:
        kwargs = {}
    signature = _UFuncSignature(input_core_dims, output_core_dims)
    if exclude_dims:
        if not isinstance(exclude_dims, set):
            raise TypeError(f"Expected exclude_dims to be a 'set'. Received '{type(exclude_dims).__name__}' instead.")
        if not exclude_dims <= signature.all_core_dims:
            raise ValueError(f'each dimension in `exclude_dims` must also be a core dimension in the function signature. Please make {exclude_dims - signature.all_core_dims} a core dimension')
    if dask == 'parallelized':
        if dask_gufunc_kwargs is None:
            dask_gufunc_kwargs = {}
        else:
            dask_gufunc_kwargs = dask_gufunc_kwargs.copy()
        if meta is not None:
            warnings.warn('``meta`` should be given in the ``dask_gufunc_kwargs`` parameter. It will be removed as direct parameter in a future version.', FutureWarning, stacklevel=2)
            dask_gufunc_kwargs.setdefault('meta', meta)
        if output_sizes is not None:
            warnings.warn('``output_sizes`` should be given in the ``dask_gufunc_kwargs`` parameter. It will be removed as direct parameter in a future version.', FutureWarning, stacklevel=2)
            dask_gufunc_kwargs.setdefault('output_sizes', output_sizes)
    if kwargs:
        func = functools.partial(func, **kwargs)
    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=False)
    if isinstance(keep_attrs, bool):
        keep_attrs = 'override' if keep_attrs else 'drop'
    variables_vfunc = functools.partial(apply_variable_ufunc, func, signature=signature, exclude_dims=exclude_dims, keep_attrs=keep_attrs, dask=dask, vectorize=vectorize, output_dtypes=output_dtypes, dask_gufunc_kwargs=dask_gufunc_kwargs)
    if any((isinstance(a, GroupBy) for a in args)):
        this_apply = functools.partial(apply_ufunc, func, input_core_dims=input_core_dims, output_core_dims=output_core_dims, exclude_dims=exclude_dims, join=join, dataset_join=dataset_join, dataset_fill_value=dataset_fill_value, keep_attrs=keep_attrs, dask=dask, vectorize=vectorize, output_dtypes=output_dtypes, dask_gufunc_kwargs=dask_gufunc_kwargs)
        return apply_groupby_func(this_apply, *args)
    elif any((is_dict_like(a) for a in args)):
        return apply_dataset_vfunc(variables_vfunc, *args, signature=signature, join=join, exclude_dims=exclude_dims, dataset_join=dataset_join, fill_value=dataset_fill_value, keep_attrs=keep_attrs, on_missing_core_dim=on_missing_core_dim)
    elif any((isinstance(a, DataArray) for a in args)):
        return apply_dataarray_vfunc(variables_vfunc, *args, signature=signature, join=join, exclude_dims=exclude_dims, keep_attrs=keep_attrs)
    elif any((isinstance(a, Variable) for a in args)):
        return variables_vfunc(*args)
    else:
        return apply_array_ufunc(func, *args, dask=dask)