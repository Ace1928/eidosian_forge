from ase.atoms import Atoms
def get_s22x5_distance(name, dist=None):
    """Returns the relative intermolecular distance in angstroms.
       Values are in Angstrom and are relative to the original s22 distance.
    """
    s22_, s22x5_, s22_name, dist_ = identify_s22_sys(name, dist)
    if s22_ is True:
        raise KeyError('System must be in s22x5')
    else:
        x00 = data[s22_name]['positions 1.0'][0][0]
        x01 = data[s22_name]['positions 1.0'][-1][0]
        x10 = data[s22_name]['positions ' + dist_][0][0]
        x11 = data[s22_name]['positions ' + dist_][-1][0]
        d0 = x01 - x00
        d1 = x11 - x10
        return d1 - d0