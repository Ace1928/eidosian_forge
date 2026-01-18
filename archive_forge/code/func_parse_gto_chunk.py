import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def parse_gto_chunk(chunk):
    atoms = None
    forces = None
    energy = None
    dipole = None
    quadrupole = None
    for theory, pattern in _e_gto.items():
        matches = pattern.findall(chunk)
        if matches:
            energy = float(matches[-1].replace('D', 'E')) * Hartree
            break
    gradblocks = _gto_grad.findall(chunk)
    if gradblocks:
        gradblock = gradblocks[-1].strip().split('\n')
        natoms = len(gradblock)
        symbols = []
        pos = np.zeros((natoms, 3))
        forces = np.zeros((natoms, 3))
        for i, line in enumerate(gradblock):
            line = line.strip().split()
            symbols.append(line[1])
            pos[i] = [float(x) for x in line[2:5]]
            forces[i] = [-float(x) for x in line[5:8]]
        pos *= Bohr
        forces *= Hartree / Bohr
        atoms = Atoms(symbols, positions=pos)
    dipole, quadrupole = _get_multipole(chunk)
    kpts = _get_gto_kpts(chunk)
    if atoms is None:
        atoms = _parse_geomblock(chunk)
    if atoms is None:
        return
    calc = SinglePointDFTCalculator(atoms=atoms, energy=energy, free_energy=energy, forces=forces, dipole=dipole)
    calc.kpts = kpts
    atoms.calc = calc
    return atoms