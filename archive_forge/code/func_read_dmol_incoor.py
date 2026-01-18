from datetime import datetime
import numpy as np
from ase import Atom, Atoms
from ase.geometry.cell import cell_to_cellpar, cellpar_to_cell
from ase.units import Bohr
from ase.utils import reader, writer
@reader
def read_dmol_incoor(fd, bohr=True):
    """ Reads an incoor file and returns an atoms object.

    Notes
    -----
    If bohr is True then incoor is assumed to be in bohr and the data
    is rescaled to Angstrom.
    """
    lines = fd.readlines()
    symbols = []
    positions = []
    for i, line in enumerate(lines):
        if line.startswith('$cell vectors'):
            cell = np.zeros((3, 3))
            for j, line in enumerate(lines[i + 1:i + 4]):
                cell[j, :] = [float(fld) for fld in line.split()]
        if line.startswith('$coordinates'):
            j = i + 1
            while True:
                if lines[j].startswith('$end'):
                    break
                flds = lines[j].split()
                symbols.append(flds[0])
                positions.append(flds[1:4])
                j += 1
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    if bohr:
        atoms.cell = atoms.cell * Bohr
        atoms.positions = atoms.positions * Bohr
    return atoms