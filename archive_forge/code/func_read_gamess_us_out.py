import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def read_gamess_us_out(fd):
    atoms = None
    energy = None
    forces = None
    dipole = None
    for line in fd:
        if _geom_re.match(line):
            fd.readline()
            symbols = []
            pos = []
            while True:
                atom = _atom_re.match(fd.readline())
                if atom is None:
                    break
                symbol, _, x, y, z = atom.groups()
                symbols.append(symbol.capitalize())
                pos.append(list(map(float, [x, y, z])))
            atoms = Atoms(symbols, np.array(pos) * Bohr)
            continue
        ematch = _energy_re.match(line)
        if ematch is not None:
            energy = float(ematch.group(1)) * Hartree
        elif line.strip().startswith('TOTAL ENERGY'):
            energy = float(line.strip().split()[-1]) * Hartree
        elif line.strip().startswith('THE FOLLOWING METHOD AND ENERGY'):
            energy = float(fd.readline().strip().split()[-1]) * Hartree
        elif _grad_re.match(line):
            for _ in range(3):
                fd.readline()
            grad = []
            while True:
                atom = _atom_re.match(fd.readline())
                if atom is None:
                    break
                grad.append(list(map(float, atom.groups()[2:])))
            forces = -np.array(grad) * Hartree / Bohr
        elif _dipole_re.match(line):
            dipole = np.array(list(map(float, fd.readline().split()[:3])))
            dipole *= Debye
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, dipole=dipole)
    return atoms