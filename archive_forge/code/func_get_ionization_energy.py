from ase.symbols import string2symbols
def get_ionization_energy(name, vertical=True):
    """Return the experimental ionization energy from the database.

    If vertical is True, the vertical ionization energy is returned if
    available.
    """
    if name not in data:
        raise KeyError('System %s not in database.' % name)
    elif 'ionization energy' not in data[name]:
        raise KeyError('No data on ionization energy for system %s.' % name)
    elif vertical and 'vertical ionization energy' in data[name]:
        return data[name]['vertical ionization energy']
    else:
        return data[name]['ionization energy']