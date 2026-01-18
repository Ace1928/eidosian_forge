import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.calculators.calculator import InputError, Calculator
from ase.calculators.gaussian import Gaussian
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses_iupac2016, chemical_symbols
from ase.io import ParseError
from ase.io.zmatrix import parse_zmatrix
from ase.units import Bohr, Hartree
def read_gaussian_out(fd, index=-1):
    configs = []
    atoms = None
    energy = None
    dipole = None
    forces = None
    for line in fd:
        line = line.strip()
        if line.startswith('1\\1\\GINC'):
            break
        if line == 'Input orientation:' or line == 'Z-Matrix orientation:':
            if atoms is not None:
                atoms.calc = SinglePointCalculator(atoms, energy=energy, dipole=dipole, forces=forces)
                _compare_merge_configs(configs, atoms)
            atoms = None
            energy = None
            dipole = None
            forces = None
            numbers = []
            positions = []
            pbc = np.zeros(3, dtype=bool)
            cell = np.zeros((3, 3))
            npbc = 0
            for _ in range(4):
                fd.readline()
            while True:
                match = _re_atom.match(fd.readline())
                if match is None:
                    break
                number = int(match.group(1))
                pos = list(map(float, match.group(2, 3, 4)))
                if number == -2:
                    pbc[npbc] = True
                    cell[npbc] = pos
                    npbc += 1
                else:
                    numbers.append(max(number, 0))
                    positions.append(pos)
            atoms = Atoms(numbers, positions, pbc=pbc, cell=cell)
        elif line.startswith('Energy=') or line.startswith('SCF Done:'):
            energy = float(line.split('=')[1].split()[0].replace('D', 'e'))
            energy *= Hartree
        elif line.startswith('E2 =') or line.startswith('E3 =') or line.startswith('E4(') or line.startswith('DEMP5 =') or line.startswith('E2('):
            energy = float(line.split('=')[-1].strip().replace('D', 'e'))
            energy *= Hartree
        elif line.startswith('Wavefunction amplitudes converged. E(Corr)'):
            energy = float(line.split('=')[-1].strip().replace('D', 'e'))
            energy *= Hartree
        elif _re_l716.match(line):
            line = fd.readline().strip()
            if not line.startswith('Dipole'):
                continue
            dip = line.split('=')[1].replace('D', 'e')
            tokens = dip.split()
            dipole = []
            if len(tokens) == 3:
                dipole = list(map(float, tokens))
            elif len(dip) % 3 == 0:
                nchars = len(dip) // 3
                for i in range(3):
                    dipole.append(float(dip[nchars * i:nchars * (i + 1)]))
            else:
                dipole = None
                continue
            dipole = np.array(dipole) * Bohr
        elif _re_forceblock.match(line):
            fd.readline()
            fd.readline()
            forces = []
            while True:
                match = _re_atom.match(fd.readline())
                if match is None:
                    break
                forces.append(list(map(float, match.group(2, 3, 4))))
            forces = np.array(forces) * Hartree / Bohr
    if atoms is not None:
        atoms.calc = SinglePointCalculator(atoms, energy=energy, dipole=dipole, forces=forces)
        _compare_merge_configs(configs, atoms)
    return configs[index]