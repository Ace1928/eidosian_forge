from collections import OrderedDict
from functools import reduce, partial
from itertools import chain
from operator import attrgetter, mul
import math
import warnings
from ..units import (
from ..util.pyutil import deprecated
from ..util._expr import Expr, Symbol
from .rates import RateExpr, MassAction
@deprecated(last_supported_version='0.5.3', will_be_missing_in='0.8.0', use_instead='pyodesys.chained_parameter_variation')
def chained_parameter_variation(odesys, durations, init_conc, varied_params, default_params, integrate_kwargs=None):
    """Integrate an ODE-system for a serie of durations with some parameters changed in-between

    Parameters
    ----------
    odesys : :class:`pyodesys.ODESys` instance
    durations : iterable of floats
    init_conc : dict or array_like
    varied_params : dict mapping parameter name to array_like
        Each array_like need to be of same length as durations.
    default_params : dict or array_like
        Default values for the parameters of the ODE system.
    integrate_kwargs : dict
        Keyword arguments passed on to :meth:`pyodesys.ODESys.integrate`.

    """
    for k, v in varied_params.items():
        if len(v) != len(durations):
            raise ValueError('Mismathced lengths of durations and varied_params')
    integrate_kwargs = integrate_kwargs or {}
    touts = []
    couts = []
    infos = {}
    c0 = init_conc.copy()
    for idx, duration in enumerate(durations):
        params = default_params.copy()
        for k, v in varied_params.items():
            params[k] = v[idx]
        tout, cout, info = odesys.integrate(duration, c0, params, **integrate_kwargs)
        c0 = cout[-1, :]
        idx0 = 0 if idx == 0 else 1
        t_global = 0 if idx == 0 else touts[-1][-1]
        touts.append(tout[idx0:] + t_global)
        couts.append(cout[idx0:, ...])
        for k, v in info.items():
            if k.startswith('internal'):
                continue
            if k in infos:
                infos[k] += (v,)
            else:
                infos[k] = (v,)
    return (np.concatenate(touts), np.concatenate(couts), infos)