from typing import Dict as tDict, Set as tSet
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.quantities import Quantity
from .dimensions import Dimension
@staticmethod
def get_unit_system(unit_system):
    if isinstance(unit_system, UnitSystem):
        return unit_system
    if unit_system not in UnitSystem._unit_systems:
        raise ValueError('Unit system is not supported. Currentlysupported unit systems are {}'.format(', '.join(sorted(UnitSystem._unit_systems))))
    return UnitSystem._unit_systems[unit_system]