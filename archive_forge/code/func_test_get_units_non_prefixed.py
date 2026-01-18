from sympy.physics.units import DimensionSystem, joule, second, ampere
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.physics.units.definitions import c, kg, m, s
from sympy.physics.units.definitions.dimension_definitions import length, time
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.physics.units.util import convert_to
def test_get_units_non_prefixed():
    from sympy.physics.units import volt, ohm
    unit_system = UnitSystem.get_unit_system('SI')
    units = unit_system.get_units_non_prefixed()
    for prefix in ['giga', 'tera', 'peta', 'exa', 'zetta', 'yotta', 'kilo', 'hecto', 'deca', 'deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto']:
        for unit in units:
            assert isinstance(unit, Quantity), f'{unit} must be a Quantity, not {type(unit)}'
            assert not unit.is_prefixed, f'{unit} is marked as prefixed'
            assert not unit.is_physical_constant, f'{unit} is marked as physics constant'
            assert not unit.name.name.startswith(prefix), f'Unit {unit.name} has prefix {prefix}'
    assert volt in units
    assert ohm in units