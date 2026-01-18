import os
import numpy as np
from ase.calculators.calculator import (FileIOCalculator, kpts2ndarray,
from ase.units import Hartree, Bohr
def read_charges_energy_dipole(self):
    """Get partial charges on atoms
            in case we cannot find charges they are set to None
        """
    with open(os.path.join(self.directory, 'detailed.out'), 'r') as fd:
        lines = fd.readlines()
    for line in lines:
        if line.strip().startswith('Total energy:'):
            energy = float(line.split()[2]) * Hartree
            break
    qm_charges = []
    for n, line in enumerate(lines):
        if 'Atom' and 'Charge' in line:
            chargestart = n + 1
            break
    else:
        return (None, energy, None)
    lines1 = lines[chargestart:chargestart + len(self.atoms)]
    for line in lines1:
        qm_charges.append(float(line.split()[-1]))
    dipole = None
    for line in lines:
        if 'Dipole moment:' in line and 'au' in line:
            words = line.split()
            dipole = np.array([float(w) for w in words[-4:-1]]) * Bohr
    return (np.array(qm_charges), energy, dipole)