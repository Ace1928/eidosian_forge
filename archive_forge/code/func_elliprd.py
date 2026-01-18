from .functions import defun, defun_wrapped
@defun
def elliprd(ctx, x, y, z):
    """
    Evaluates the degenerate Carlson symmetric elliptic integral
    of the third kind or Carlson elliptic integral of the
    second kind `R_D(x,y,z) = R_J(x,y,z,z)`.

    See :func:`~mpmath.elliprj` for additional information.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprd(1,2,3)
        0.2904602810289906442326534
        >>> elliprj(1,2,3,3)
        0.2904602810289906442326534

    The so-called *second lemniscate constant*, a transcendental number::

        >>> elliprd(0,2,1)/3
        0.5990701173677961037199612
        >>> extradps(25)(quad)(lambda t: t**2/sqrt(1-t**4), [0,1])
        0.5990701173677961037199612
        >>> gamma('3/4')**2/sqrt(2*pi)
        0.5990701173677961037199612

    """
    return ctx.elliprj(x, y, z, z)