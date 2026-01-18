import logging
from pyomo.core.base.units_container import units, UnitsError
from pyomo.core.base import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port, Arc
from pyomo.mpec import Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.numvalue import native_types
from pyomo.util.components import iter_component
from pyomo.common.collections import ComponentSet
def assert_units_equivalent(*args):
    """
    Raise an exception if the units are inconsistent within an
    expression, or not equivalent across all the passed
    expressions.

    Parameters
    ----------
    args : an argument list of Pyomo expressions
        The Pyomo expressions to test

    Raises
    ------
    :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`
    """
    pint_units = [units._get_pint_units(arg) for arg in args]
    pint_unit_compare = pint_units[0]
    for pint_unit in pint_units:
        if not units._equivalent_pint_units(pint_unit_compare, pint_unit):
            raise UnitsError('Units between {} and {} are not consistent.'.format(str(pint_unit_compare), str(pint_unit)))