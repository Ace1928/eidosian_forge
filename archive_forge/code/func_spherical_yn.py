import cupy
from cupy import _core
def spherical_yn(n, z, derivative=False):
    """Spherical Bessel function of the second kind or its derivative.

    Parameters
    ----------
    n : cupy.ndarray
        Order of the Bessel function.
    z : cupy.ndarray
        Argument of the Bessel function.
        Real-valued input.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    yn : cupy.ndarray

    See Also
    -------
    :func:`scipy.special.spherical_yn`

    """
    if cupy.iscomplexobj(z):
        if derivative:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif derivative:
        return _spherical_dyn_real(n, z)
    else:
        return _spherical_yn_real(n, z)