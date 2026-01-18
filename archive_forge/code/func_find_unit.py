from .dimensions import Dimension, DimensionSystem
from .unitsystem import UnitSystem
from .util import convert_to
from .quantities import Quantity
from .definitions.dimension_definitions import (
from .prefixes import (
from .definitions import (
from .systems import (
def find_unit(quantity, unit_system='SI'):
    """
    Return a list of matching units or dimension names.

    - If ``quantity`` is a string -- units/dimensions containing the string
    `quantity`.
    - If ``quantity`` is a unit or dimension -- units having matching base
    units or dimensions.

    Examples
    ========

    >>> from sympy.physics import units as u
    >>> u.find_unit('charge')
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit(u.charge)
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit("ampere")
    ['ampere', 'amperes']
    >>> u.find_unit('angstrom')
    ['angstrom', 'angstroms']
    >>> u.find_unit('volt')
    ['volt', 'volts', 'electronvolt', 'electronvolts', 'planck_voltage']
    >>> u.find_unit(u.inch**3)[:9]
    ['L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter']
    """
    unit_system = UnitSystem.get_unit_system(unit_system)
    import sympy.physics.units as u
    rv = []
    if isinstance(quantity, str):
        rv = [i for i in dir(u) if quantity in i and isinstance(getattr(u, i), Quantity)]
        dim = getattr(u, quantity)
        if isinstance(dim, Dimension):
            rv.extend(find_unit(dim))
    else:
        for i in sorted(dir(u)):
            other = getattr(u, i)
            if not isinstance(other, Quantity):
                continue
            if isinstance(quantity, Quantity):
                if quantity.dimension == other.dimension:
                    rv.append(str(i))
            elif isinstance(quantity, Dimension):
                if other.dimension == quantity:
                    rv.append(str(i))
            elif other.dimension == Dimension(unit_system.get_dimensional_expr(quantity)):
                rv.append(str(i))
    return sorted(set(rv), key=lambda x: (len(x), x))