import pytest
import numpy as np
from ase.build import bulk, make_supercell
from ase.lattice import FCC, BCC
from ase.calculators.emt import EMT
def getenergy(atoms, eref=None):
    atoms.calc = EMT()
    e = atoms.get_potential_energy() / len(atoms)
    if eref is not None:
        err = abs(e - eref)
        print('natoms', len(atoms), 'err', err)
        assert err < 1e-12, err
    return e