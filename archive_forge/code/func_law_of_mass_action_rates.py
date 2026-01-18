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
def law_of_mass_action_rates(conc, rsys, variables=None):
    """Returns a generator of reaction rate expressions

    Rates from the law of mass action (:attr:`Reaction.inact_reac` ignored)
    from a :class:`ReactionSystem`.

    Parameters
    ----------
    conc : array_like
        concentrations (floats or symbolic objects)
    rsys : ReactionSystem instance
        See :class:`ReactionSystem`
    variables : dict (optional)
        to override parameters in the rate expressions of the reactions

    Examples
    --------
    >>> from chempy import ReactionSystem, Reaction
    >>> line, keys = 'H2O -> H+ + OH- ; 1e-4', 'H2O H+ OH-'
    >>> rxn = Reaction.from_string(line, keys)
    >>> rsys = ReactionSystem([rxn], keys)
    >>> next(law_of_mass_action_rates([55.4, 1e-7, 1e-7], rsys))
    0.00554
    >>> from chempy.kinetics.rates import Arrhenius, MassAction
    >>> rxn.param = MassAction(Arrhenius({'A': 1e10, 'Ea_over_R': 9314}))
    >>> print('%.5g' % next(law_of_mass_action_rates([55.4, 1e-7, 1e-7], rsys, {'temperature': 293})))
    0.0086693

    """
    for idx_r, rxn in enumerate(rsys.rxns):
        if isinstance(rxn.param, RateExpr):
            if isinstance(rxn.param, MassAction):
                yield rxn.param(dict(chain(variables.items(), zip(rsys.substances.keys(), conc))), reaction=rxn)
            else:
                raise ValueError('Not mass-action rate in reaction %d' % idx_r)
        else:
            rate = 1
            for substance_key, coeff in rxn.reac.items():
                s_idx = rsys.as_substance_index(substance_key)
                rate *= conc[s_idx] ** coeff
            yield (rate * rxn.param)