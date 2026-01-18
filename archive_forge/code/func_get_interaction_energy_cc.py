from ase.atoms import Atoms
def get_interaction_energy_cc(name, dist=None):
    """Returns the S22/S26 CCSD(T)/CBS CP interaction energy in eV.
    """
    s22_, s22x5_, s22_name, dist_ = identify_s22_sys(name, dist)
    return data[s22_name]['interaction energy CC']