import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def labeled_comprehension(input, labels, index, func, out_dtype, default, pass_positions=False):
    """Array resulting from applying ``func`` to each labeled region.

    Roughly equivalent to [func(input[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like input)
    to subsets of an N-D image array specified by `labels` and `index`.
    The option exists to provide the function with positional parameters as the
    second argument.

    Args:
        input (cupy.ndarray): Data from which to select `labels` to process.
        labels (cupy.ndarray or None):  Labels to objects in `input`. If not
            None, array must be same shape as `input`. If None, `func` is
            applied to raveled `input`.
        index (int, sequence of ints or None): Subset of `labels` to which to
            apply `func`. If a scalar, a single value is returned. If None,
            `func` is applied to all non-zero values of `labels`.
        func (callable): Python function to apply to `labels` from `input`.
        out_dtype (dtype): Dtype to use for `result`.
        default (int, float or None): Default return value when a element of
            `index` does not exist in `labels`.
        pass_positions (bool, optional): If True, pass linear indices to `func`
            as a second argument.

    Returns:
        cupy.ndarray: Result of applying `func` to each of `labels` to `input`
        in `index`.

    .. seealso:: :func:`scipy.ndimage.labeled_comprehension`
    """
    as_scalar = cupy.isscalar(index)
    input = cupy.asarray(input)
    if pass_positions:
        positions = cupy.arange(input.size).reshape(input.shape)
    if labels is None:
        if index is not None:
            raise ValueError('index without defined labels')
        if not pass_positions:
            return func(input.ravel())
        else:
            return func(input.ravel(), positions.ravel())
    try:
        input, labels = cupy.broadcast_arrays(input, labels)
    except ValueError:
        raise ValueError('input and labels must have the same shape (excepting dimensions with width 1)')
    if index is None:
        if not pass_positions:
            return func(input[labels > 0])
        else:
            return func(input[labels > 0], positions[labels > 0])
    index = cupy.atleast_1d(index)
    if cupy.any(index.astype(labels.dtype).astype(index.dtype) != index):
        raise ValueError('Cannot convert index values from <%s> to <%s> (labels.dtype) without loss of precision' % (index.dtype, labels.dtype))
    index = index.astype(labels.dtype)
    lo = index.min()
    hi = index.max()
    mask = (labels >= lo) & (labels <= hi)
    labels = labels[mask]
    input = input[mask]
    if pass_positions:
        positions = positions[mask]
    label_order = labels.argsort()
    labels = labels[label_order]
    input = input[label_order]
    if pass_positions:
        positions = positions[label_order]
    index_order = index.argsort()
    sorted_index = index[index_order]

    def do_map(inputs, output):
        """labels must be sorted"""
        nidx = sorted_index.size
        lo = cupy.searchsorted(labels, sorted_index, side='left')
        hi = cupy.searchsorted(labels, sorted_index, side='right')
        for i, low, high in zip(range(nidx), lo, hi):
            if low == high:
                continue
            output[i] = func(*[inp[low:high] for inp in inputs])
    if out_dtype == object:
        temp = {i: default for i in range(index.size)}
    else:
        temp = cupy.empty(index.shape, out_dtype)
        if default is None and temp.dtype.kind in 'fc':
            default = numpy.nan
        temp[:] = default
    if not pass_positions:
        do_map([input], temp)
    else:
        do_map([input, positions], temp)
    if out_dtype == object:
        index_order = cupy.asnumpy(index_order)
        output = [temp[i] for i in index_order.argsort()]
    else:
        output = cupy.zeros(index.shape, out_dtype)
        output[cupy.asnumpy(index_order)] = temp
    if as_scalar:
        output = output[0]
    return output