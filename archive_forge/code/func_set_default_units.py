import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
def set_default_units(system=None, currency=None, current=None, information=None, length=None, luminous_intensity=None, mass=None, substance=None, temperature=None, time=None):
    """
    Set the default units in which simplified quantities will be
    expressed.

    system sets the unit system, and can be "SI" or "cgs". All other
    keyword arguments will accept either a string or a unit quantity.
    An error will be raised if it is not possible to convert between
    old and new defaults, so it is not possible to set "kg" as the
    default unit for time.

    If both system and individual defaults are given, the system
    defaults will be applied first, followed by the individual ones.
    """
    if system is not None:
        system = system.lower()
        try:
            assert system in ('si', 'cgs')
        except AssertionError:
            raise ValueError('system must be "SI" or "cgs", got "%s"' % system)
        if system == 'si':
            UnitCurrent.set_default_unit('A')
            UnitLength.set_default_unit('m')
            UnitMass.set_default_unit('kg')
        elif system == 'cgs':
            UnitLength.set_default_unit('cm')
            UnitMass.set_default_unit('g')
        UnitLuminousIntensity.set_default_unit('cd')
        UnitSubstance.set_default_unit('mol')
        UnitTemperature.set_default_unit('degK')
        UnitTime.set_default_unit('s')
    UnitCurrency.set_default_unit(currency)
    UnitCurrent.set_default_unit(current)
    UnitInformation.set_default_unit(information)
    UnitLength.set_default_unit(length)
    UnitLuminousIntensity.set_default_unit(luminous_intensity)
    UnitMass.set_default_unit(mass)
    UnitSubstance.set_default_unit(substance)
    UnitTemperature.set_default_unit(temperature)
    UnitTime.set_default_unit(time)