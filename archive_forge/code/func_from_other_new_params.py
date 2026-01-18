from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@classmethod
def from_other_new_params(cls, ori, par_subs, new_pars, new_par_names=None, new_latex_par_names=None, **kwargs):
    """ Creates a new instance with an existing one as a template (with new parameters)

        Calls ``.from_other`` but first it replaces some parameters according to ``par_subs``
        and (optionally) introduces new parameters given in ``new_pars``.

        Parameters
        ----------
        ori : SymbolicSys instance
        par_subs : dict
            Dictionary with substitutions (mapping symbols to new expressions) for parameters.
            Parameters appearing in this instance will be omitted in the new instance.
        new_pars : iterable (optional)
            Iterable of symbols for new parameters.
        new_par_names : iterable of str
            Names of the new parameters given in ``new_pars``.
        new_latex_par_names : iterable of str
            TeX formatted names of the new parameters given in ``new_pars``.
        \\*\\*kwargs:
            Keyword arguments passed to ``.from_other``.

        Returns
        -------
        Intance of the class
        extra : dict with keys:
            - recalc_params : ``f(t, y, p1) -> p0``

        """
    new_exprs = [expr.subs(par_subs) for expr in ori.exprs]
    drop_idxs = [ori.params.index(par) for par in par_subs]
    params = _skip(drop_idxs, ori.params, False) + list(new_pars)
    back_substitute = _Callback(ori.indep, ori.dep, params, list(par_subs.values()), Lambdify=ori.be.Lambdify)

    def recalc_params(t, y, p):
        rev = back_substitute(t, y, p)
        return _reinsert(drop_idxs, np.repeat(np.atleast_2d(p), rev.shape[0], axis=0), rev)[..., :len(ori.params)]
    return (cls.from_other(ori, dep_exprs=zip(ori.dep, new_exprs), params=params, param_names=_skip(drop_idxs, ori.param_names, False) + list(new_par_names or []), latex_param_names=_skip(drop_idxs, ori.latex_param_names, False) + list(new_latex_par_names or []), **kwargs), {'recalc_params': recalc_params})