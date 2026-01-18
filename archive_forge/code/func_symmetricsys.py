from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def symmetricsys(dep_tr=None, indep_tr=None, SuperClass=TransformedSys, **kwargs):
    """ A factory function for creating symmetrically transformed systems.

    Creates a new subclass which applies the same transformation for each dependent variable.

    Parameters
    ----------
    dep_tr : pair of callables (default: None)
        Forward and backward transformation callbacks to be applied to the
        dependent variables.
    indep_tr : pair of callables (default: None)
        Forward and backward transformation to be applied to the
        independent variable.
    SuperClass : class
    \\*\\*kwargs :
        Default keyword arguments for the TransformedSys subclass.

    Returns
    -------
    Subclass of SuperClass (by default :class:`TransformedSys`).

    Examples
    --------
    >>> import sympy
    >>> logexp = (sympy.log, sympy.exp)
    >>> def psimp(exprs):
    ...     return [sympy.powsimp(expr.expand(), force=True) for expr in exprs]
    ...
    >>> LogLogSys = symmetricsys(logexp, logexp, exprs_process_cb=psimp)
    >>> mysys = LogLogSys.from_callback(lambda x, y, p: [-y[0], y[0] - y[1]], 2, 0)
    >>> mysys.exprs
    (-exp(x_0), -exp(x_0) + exp(x_0 + y_0 - y_1))

    """
    if dep_tr is not None:
        if not callable(dep_tr[0]) or not callable(dep_tr[1]):
            raise ValueError('Exceptected dep_tr to be a pair of callables')
    if indep_tr is not None:
        if not callable(indep_tr[0]) or not callable(indep_tr[1]):
            raise ValueError('Exceptected indep_tr to be a pair of callables')

    class _SymmetricSys(SuperClass):

        def __init__(self, dep_exprs, indep=None, **inner_kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(inner_kwargs)
            dep, exprs = zip(*dep_exprs)
            super(_SymmetricSys, self).__init__(zip(dep, exprs), indep, dep_transf=list(zip(list(map(dep_tr[0], dep)), list(map(dep_tr[1], dep)))) if dep_tr is not None else None, indep_transf=(indep_tr[0](indep), indep_tr[1](indep)) if indep_tr is not None else None, **new_kwargs)

        @classmethod
        def from_callback(cls, cb, ny=None, nparams=None, **inner_kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(inner_kwargs)
            return SuperClass.from_callback(cb, ny, nparams, dep_transf_cbs=repeat(dep_tr) if dep_tr is not None else None, indep_transf_cbs=indep_tr, **new_kwargs)
    return _SymmetricSys