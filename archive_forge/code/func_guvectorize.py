import inspect
from numba.np.ufunc import _internal
from numba.np.ufunc.parallel import ParallelUFuncBuilder, ParallelGUFuncBuilder
from numba.core.registry import DelayedRegistry
from numba.np.ufunc import dufunc
from numba.np.ufunc import gufunc
def guvectorize(*args, **kwargs):
    """guvectorize(ftylist, signature, target='cpu', identity=None, **kws)

    A decorator to create NumPy generalized-ufunc object from Numba compiled
    code.

    Args
    -----
    ftylist: iterable
        An iterable of type signatures, which are either
        function type object or a string describing the
        function type.

    signature: str
        A NumPy generalized-ufunc signature.
        e.g. "(m, n), (n, p)->(m, p)"

    identity: int, str, or None
        The identity (or unit) value for the element-wise function
        being implemented.  Allowed values are None (the default), 0, 1,
        and "reorderable".

    cache: bool
        Turns on caching.

    writable_args: tuple
        a tuple of indices of input variables that are writable.

    target: str
            A string for code generation target.  Defaults to "cpu".

    Returns
    --------

    A NumPy generalized universal-function

    Example
    -------
        @guvectorize(['void(int32[:,:], int32[:,:], int32[:,:])',
                      'void(float32[:,:], float32[:,:], float32[:,:])'],
                      '(x, y),(x, y)->(x, y)')
        def add_2d_array(a, b, c):
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    c[i, j] = a[i, j] + b[i, j]

    """
    if len(args) == 1:
        ftylist = []
        signature = args[0]
        kwargs.setdefault('is_dynamic', True)
    elif len(args) == 2:
        ftylist = args[0]
        signature = args[1]
    else:
        raise TypeError('guvectorize() takes one or two positional arguments')
    if isinstance(ftylist, str):
        ftylist = [ftylist]

    def wrap(func):
        guvec = GUVectorize(func, signature, **kwargs)
        for fty in ftylist:
            guvec.add(fty)
        if len(ftylist) > 0:
            guvec.disable_compile()
        return guvec.build_ufunc()
    return wrap