import os
import os.path as op
import subprocess
import shutil
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray
def deMon_inp_to_atoms(self, filename):
    """Routine to read deMon.inp and convert it to an atoms object."""
    with open(filename, 'r') as fd:
        lines = fd.readlines()
    for i in range(len(lines)):
        if lines[i].rfind('GEOMETRY') > -1:
            if lines[i].rfind('ANGSTROM'):
                coord_units = 'Ang'
            elif lines.rfind('Bohr'):
                coord_units = 'Bohr'
            ii = i
            break
    chemical_symbols = []
    xyz = []
    atomic_numbers = []
    masses = []
    for i in range(ii + 1, len(lines)):
        try:
            line = lines[i].split()
            if len(line) > 0:
                for symbol in ase.data.chemical_symbols:
                    found = None
                    if line[0].upper().rfind(symbol.upper()) > -1:
                        found = symbol
                        break
                    if found is not None:
                        chemical_symbols.append(found)
                    else:
                        break
                    xyz.append([float(line[1]), float(line[2]), float(line[3])])
            if len(line) > 4:
                atomic_numbers.append(int(line[4]))
            if len(line) > 5:
                masses.append(float(line[5]))
        except Exception:
            raise RuntimeError
    if coord_units == 'Bohr':
        xyz = xyz * Bohr
    natoms = len(chemical_symbols)
    atoms = ase.Atoms(symbols=chemical_symbols, positions=xyz)
    if len(atomic_numbers) == natoms:
        atoms.set_atomic_numbers(atomic_numbers)
    if len(masses) == natoms:
        atoms.set_masses(masses)
    return atoms