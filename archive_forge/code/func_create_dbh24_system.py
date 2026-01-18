from ase.atoms import Atoms
def create_dbh24_system(name, **kwargs):
    """Creates a DBH24 system.
    """
    if name not in data:
        raise NotImplementedError('System %s not in database.' % name)
    d = data[name]
    if 'magmoms' not in kwargs:
        kwargs['magmoms'] = d['magmoms']
    return Atoms(d['symbols'], d['positions'], **kwargs)