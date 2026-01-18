from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
@memoize(None)
def mk_Radiolytic(*doserate_names):
    """Create a Radiolytic rate expression

    Note that there is no mass-action dependence in the resulting
    class, i.e. the rates does not depend on any concentrations.

    Parameters
    ----------
    \\*doserate_names : str instances
        Default: ('',)


    Examples
    --------
    >>> RadiolyticAlpha = mk_Radiolytic('alpha')
    >>> RadiolyticGamma = mk_Radiolytic('gamma')
    >>> dihydrogen_alpha = RadiolyticAlpha([0.8e-7])
    >>> dihydrogen_gamma = RadiolyticGamma([0.45e-7])
    >>> RadiolyticAB = mk_Radiolytic('alpha', 'beta')

    Notes
    -----
    The instance __call__ will require by default ``'density'`` and ``'doserate'``
    in variables.

    """
    if len(doserate_names) == 0:
        doserate_names = ('',)

    class _Radiolytic(RadiolyticBase):
        argument_names = tuple(('radiolytic_yield{0}'.format('' if drn == '' else '_' + drn) for drn in doserate_names))
        parameter_keys = ('density',) + tuple(('doserate{0}'.format('' if drn == '' else '_' + drn) for drn in doserate_names))

        def args_dimensionality(self, reaction):
            N = base_registry['amount']
            E = get_derived_unit(base_registry, 'energy')
            return (dict(zip(dimension_codes, N / E)),) * self.nargs

        def g_values(self, *args, **kwargs):
            return OrderedDict(zip(self.parameter_keys[1:], self.all_args(*args, **kwargs)))

        @deprecated(use_instead='Radiolytic.all_args')
        def g_value(self, variables, backend=math, **kwargs):
            g_val, = self.all_args(variables, backend=backend, **kwargs)
            return g_val

        def __call__(self, variables, backend=math, reaction=None, **kwargs):
            return variables['density'] * reduce(add, [variables[k] * gval for k, gval in zip(self.parameter_keys[1:], self.all_args(variables, backend=backend, **kwargs))])
    _Radiolytic.__name__ = 'Radiolytic' if doserate_names == ('',) else 'Radiolytic_' + '_'.join(doserate_names)
    return _Radiolytic