def str_to_array(s):
    """
    Simplistic converter of strings from repr to float NumPy arrays.

    If the repr representation has ellipsis in it, then this will fail.

    Parameters
    ----------
    s : str
        The repr version of a NumPy array.

    Examples
    --------
    >>> s = "array([ 0.3,  inf,  nan])"
    >>> a = str_to_array(s)

    """
    import numpy as np
    from numpy import inf, nan
    if s.startswith(u'array'):
        s = s[6:-1]
    if s.startswith(u'['):
        a = np.array(eval(s), dtype=float)
    else:
        a = np.atleast_1d(float(s))
    return a