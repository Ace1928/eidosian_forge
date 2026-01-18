from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@classmethod
def from_other_new_params_by_name(cls, ori, par_subs, new_par_names=(), **kwargs):
    """ Creates a new instance with an existing one as a template (with new parameters)

        Calls ``.from_other_new_params`` but first it creates the new instances from user provided
        callbacks generating the expressions the parameter substitutions.

        Parameters
        ----------
        ori : SymbolicSys instance
        par_subs : dict mapping str to ``f(t, y{}, p{}) -> expr``
            User provided callbacks for parameter names in ``ori``.
        new_par_names : iterable of str
        \\*\\*kwargs:
            Keyword arguments passed to ``.from_other_new_params``.

        """
    if not ori.dep_by_name:
        warnings.warn('dep_by_name is not True')
    if not ori.par_by_name:
        warnings.warn('par_by_name is not True')
    dep = dict(zip(ori.names, ori.dep))
    new_pars = ori.be.real_symarray('p', len(ori.params) + len(new_par_names))[len(ori.params):]
    par = dict(chain(zip(ori.param_names, ori.params), zip(new_par_names, new_pars)))
    par_symb_subs = OrderedDict([(ori.params[ori.param_names.index(pk)], cb(ori.indep, dep, par, backend=ori.be)) for pk, cb in par_subs.items()])
    return cls.from_other_new_params(ori, par_symb_subs, new_pars, new_par_names=new_par_names, **kwargs)