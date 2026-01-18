import sys
def extract_vars(*names, **kw):
    """Extract a set of variables by name from another frame.

    Parameters
    ----------
    *names : str
        One or more variable names which will be extracted from the caller's
        frame.
    **kw : integer, optional
        How many frames in the stack to walk when looking for your variables.
        The default is 0, which will use the frame where the call was made.

    Examples
    --------
    ::

        In [2]: def func(x):
           ...:     y = 1
           ...:     print(sorted(extract_vars('x','y').items()))
           ...:

        In [3]: func('hello')
        [('x', 'hello'), ('y', 1)]
    """
    depth = kw.get('depth', 0)
    callerNS = sys._getframe(depth + 1).f_locals
    return dict(((k, callerNS[k]) for k in names))