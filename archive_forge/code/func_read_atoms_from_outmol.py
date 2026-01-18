import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def read_atoms_from_outmol(self):
    """ Reads atomic positions and cell from outmol file and returns atoms
        object.

        If no cell vectors are found in outmol the cell is set to np.eye(3) and
        pbc 000.

        Formatting for cell in outmol :
         translation vector [a0]    1    5.1    0.0    5.1
         translation vector [a0]    2    5.1    5.1    0.0
         translation vector [a0]    3    0.0    5.1    5.1

        Formatting for positions in outmol:
        df              ATOMIC  COORDINATES (au)
        df            x          y          z
        df   Si     0.0   0.0   0.0
        df   Si     1.3   3.5   2.2
        df  binding energy      -0.2309046Ha

        Returns
        -------
        atoms (Atoms object): read atoms object
        """
    lines = self._outmol_lines()
    found_cell = False
    cell = np.zeros((3, 3))
    symbols = []
    positions = []
    pattern_translation_vectors = re.compile('\\s+translation\\s+vector')
    pattern_atomic_coordinates = re.compile('df\\s+ATOMIC\\s+COORDINATES')
    for i, line in enumerate(lines):
        if pattern_translation_vectors.match(line):
            cell[int(line.split()[3]) - 1, :] = np.array([float(x) for x in line.split()[-3:]])
            found_cell = True
        if pattern_atomic_coordinates.match(line):
            for ind, j in enumerate(range(i + 2, i + 2 + len(self.atoms))):
                flds = lines[j].split()
                symbols.append(flds[1])
                positions.append(flds[2:5])
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
    atoms.positions *= Bohr
    atoms.cell *= Bohr
    if found_cell:
        atoms.pbc = [True, True, True]
        atoms.wrap()
    else:
        atoms.pbc = [False, False, False]
    return atoms