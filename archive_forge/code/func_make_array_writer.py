import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def make_array_writer(data, out_type, has_slope=True, has_intercept=True, **kwargs):
    """Make array writer instance for array `data` and output type `out_type`

    Parameters
    ----------
    data : array-like
        array for which to create array writer
    out_type : dtype-like
        input to numpy dtype to specify array writer output type
    has_slope : {True, False}
        If True, array write can use scaling to adapt the array to `out_type`
    has_intercept : {True, False}
        If True, array write can use intercept to adapt the array to `out_type`
    \\*\\*kwargs : other keyword arguments
        to pass to the arraywriter class

    Returns
    -------
    writer : arraywriter instance
        Instance of array writer, with class adapted to `has_intercept` and
        `has_slope`.

    Examples
    --------
    >>> aw = make_array_writer(np.arange(10), np.uint8, True, True)
    >>> type(aw) == SlopeInterArrayWriter
    True
    >>> aw = make_array_writer(np.arange(10), np.uint8, True, False)
    >>> type(aw) == SlopeArrayWriter
    True
    >>> aw = make_array_writer(np.arange(10), np.uint8, False, False)
    >>> type(aw) == ArrayWriter
    True
    """
    data = np.asarray(data)
    if has_intercept and (not has_slope):
        raise ValueError('Cannot handle intercept without slope')
    if has_intercept:
        return SlopeInterArrayWriter(data, out_type, **kwargs)
    if has_slope:
        return SlopeArrayWriter(data, out_type, **kwargs)
    return ArrayWriter(data, out_type, **kwargs)