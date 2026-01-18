from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def get_derived_unit(registry, key):
    """Get the unit of a physical quantity in a provided unit system.

    Parameters
    ----------
    registry: dict (str: unit)
        mapping 'length', 'mass', 'time', 'current', 'temperature',
        'luminous_intensity', 'amount'. If registry is ``None`` the
        function returns 1.0 unconditionally.
    key: str
        one of the registry keys or one of: 'diffusivity', 'electricalmobility',
        'permittivity', 'charge', 'energy', 'concentration', 'density',
        'radiolytic_yield'.

    Examples
    --------
    >>> m, s = default_units.meter, default_units.second
    >>> get_derived_unit(SI_base_registry, 'diffusivity') == m**2/s
    True

    """
    if registry is None:
        return 1.0
    derived = {'diffusivity': registry['length'] ** 2 / registry['time'], 'electrical_mobility': registry['current'] * registry['time'] ** 2 / registry['mass'], 'permittivity': registry['current'] ** 2 * registry['time'] ** 4 / (registry['length'] ** 3 * registry['mass']), 'charge': registry['current'] * registry['time'], 'energy': registry['mass'] * registry['length'] ** 2 / registry['time'] ** 2, 'concentration': registry['amount'] / registry['length'] ** 3, 'density': registry['mass'] / registry['length'] ** 3}
    derived['diffusion'] = derived['diffusivity']
    derived['radiolytic_yield'] = registry['amount'] / derived['energy']
    derived['doserate'] = derived['energy'] / registry['mass'] / registry['time']
    derived['linear_energy_transfer'] = derived['energy'] / registry['length']
    try:
        return derived[key]
    except KeyError:
        return registry[key]