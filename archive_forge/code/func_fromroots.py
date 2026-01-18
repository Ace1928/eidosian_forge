import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
@classmethod
def fromroots(cls, roots, domain=[], window=None, symbol='x'):
    """Return series instance that has the specified roots.

        Returns a series representing the product
        ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a
        list of roots.

        Parameters
        ----------
        roots : array_like
            List of roots.
        domain : {[], None, array_like}, optional
            Domain for the resulting series. If None the domain is the
            interval from the smallest root to the largest. If [] the
            domain is the class domain. The default is [].
        window : {None, array_like}, optional
            Window for the returned series. If None the class window is
            used. The default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            Series with the specified roots.

        """
    [roots] = pu.as_series([roots], trim=False)
    if domain is None:
        domain = pu.getdomain(roots)
    elif type(domain) is list and len(domain) == 0:
        domain = cls.domain
    if window is None:
        window = cls.window
    deg = len(roots)
    off, scl = pu.mapparms(domain, window)
    rnew = off + scl * roots
    coef = cls._fromroots(rnew) / scl ** deg
    return cls(coef, domain=domain, window=window, symbol=symbol)