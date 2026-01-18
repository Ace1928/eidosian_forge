import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    """
    Array-checking restype/argtypes.

    An ndpointer instance is used to describe an ndarray in restypes
    and argtypes specifications.  This approach is more flexible than
    using, for example, ``POINTER(c_double)``, since several restrictions
    can be specified, which are verified upon calling the ctypes function.
    These include data type, number of dimensions, shape and flags.  If a
    given array does not satisfy the specified restrictions,
    a ``TypeError`` is raised.

    Parameters
    ----------
    dtype : data-type, optional
        Array data-type.
    ndim : int, optional
        Number of array dimensions.
    shape : tuple of ints, optional
        Array shape.
    flags : str or tuple of str
        Array flags; may be one or more of:

          - C_CONTIGUOUS / C / CONTIGUOUS
          - F_CONTIGUOUS / F / FORTRAN
          - OWNDATA / O
          - WRITEABLE / W
          - ALIGNED / A
          - WRITEBACKIFCOPY / X

    Returns
    -------
    klass : ndpointer type object
        A type object, which is an ``_ndtpr`` instance containing
        dtype, ndim, shape and flags information.

    Raises
    ------
    TypeError
        If a given array does not satisfy the specified restrictions.

    Examples
    --------
    >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
    ...                                                  ndim=1,
    ...                                                  flags='C_CONTIGUOUS')]
    ... #doctest: +SKIP
    >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
    ... #doctest: +SKIP

    """
    if dtype is not None:
        dtype = _dtype(dtype)
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except Exception as e:
                raise TypeError('invalid flags specification') from e
            num = _num_fromflags(flags)
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)
    cache_key = (dtype, ndim, shape, num)
    try:
        return _pointer_type_cache[cache_key]
    except KeyError:
        pass
    if dtype is None:
        name = 'any'
    elif dtype.names is not None:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += '_%dd' % ndim
    if shape is not None:
        name += '_' + 'x'.join((str(x) for x in shape))
    if flags is not None:
        name += '_' + '_'.join(flags)
    if dtype is not None and shape is not None:
        base = _concrete_ndptr
    else:
        base = _ndptr
    klass = type('ndpointer_%s' % name, (base,), {'_dtype_': dtype, '_shape_': shape, '_ndim_': ndim, '_flags_': num})
    _pointer_type_cache[cache_key] = klass
    return klass