from ase.atoms import Atoms
def create_s22_system(name, dist=None, **kwargs):
    """Create S22/S26/s22x5 system.
    """
    s22_, s22x5_, s22_name, dist = identify_s22_sys(name, dist)
    if s22_ is True:
        d = data[s22_name]
        return Atoms(d['symbols'], d['positions'], **kwargs)
    elif s22x5_ is True:
        d = data[s22_name]
        pos = 'positions ' + dist
        return Atoms(d['symbols'], d[pos], **kwargs)
    else:
        raise NotImplementedError('s22/s26/s22x5 creation failed')