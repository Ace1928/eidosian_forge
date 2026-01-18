from ase.symbols import string2symbols
def get_atomization_energy(name):
    """Determine extrapolated experimental atomization energy from the database.

    The atomization energy is extrapolated from experimental heats of
    formation at room temperature, using calculated zero-point energies
    and thermal corrections.

    The atomization energy is returned in kcal/mol = 43.36 meV:

    >>> from ase.units import *; print kcal / mol
    0.0433641146392

    """
    d = data[name]
    e = d['enthalpy']
    z = d['ZPE']
    dh = d['thermal correction']
    ae = -e + z + dh
    for a in string2symbols(d['symbols']):
        h = data[a]['enthalpy']
        dh = data[a]['thermal correction']
        ae += h - dh
    return ae