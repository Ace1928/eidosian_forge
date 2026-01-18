from ase.atoms import Atoms
def get_interaction_energy_s22x5(name, dist=None, correct_offset=True):
    """Returns the S22x5 CCSD(T)/CBS CP interaction energy in eV.
    """
    s22_, s22x5_, s22_name, dist_ = identify_s22_sys(name, dist)
    if dist_ == '0.9':
        i = 0
    elif dist_ == '1.0':
        i = 1
    elif dist_ == '1.2':
        i = 2
    elif dist_ == '1.5':
        i = 3
    elif dist_ == '2.0':
        i = 4
    else:
        raise KeyError('error, mate!')
    e = data[s22_name]['interaction energies s22x5'][i]
    if correct_offset == True:
        e *= data[s22_name]['interaction energy CC'] / data[s22_name]['interaction energies s22x5'][1]
    return e