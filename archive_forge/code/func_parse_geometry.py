import numpy as np
import ase.units
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.data import chemical_symbols
def parse_geometry(filename):
    """Read atoms geometry from ACE-Molecule log file and put it to self.data.
    Parameters
    ==========
    filename: ACE-Molecule log file.

    Returns
    =======
    Dictionary of parsed geometry data.
    retval["Atomic_numbers"]: list of atomic numbers
    retval["Positions"]: list of [x, y, z] coordinates for each atoms.
    """
    with open(filename, 'r') as fd:
        lines = fd.readlines()
        start_line = 0
        end_line = 0
        for i, line in enumerate(lines):
            if line == '====================  Atoms  =====================\n':
                start_line = i
            if start_line != 0 and len(line.split('=')) > 3:
                end_line = i
                if not start_line == end_line:
                    break
        atoms = []
        positions = []
        for i in range(start_line + 1, end_line):
            atomic_number = lines[i].split()[0]
            atoms.append(str(chemical_symbols[int(atomic_number)]))
            xyz = [float(n) for n in lines[i].split()[1:4]]
            positions.append(xyz)
        return {'Atomic_numbers': atoms, 'Positions': positions}