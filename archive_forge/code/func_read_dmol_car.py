from datetime import datetime
import numpy as np
from ase import Atom, Atoms
from ase.geometry.cell import cell_to_cellpar, cellpar_to_cell
from ase.units import Bohr
from ase.utils import reader, writer
@reader
def read_dmol_car(fd):
    """ Read a dmol car-file and return an Atoms object.

    Notes
    -----
    Cell is constructed from cellpar so orientation of cell might be off.
    """
    lines = fd.readlines()
    atoms = Atoms()
    start_line = 4
    if lines[1][4:6] == 'ON':
        start_line += 1
        cell_dat = np.array([float(fld) for fld in lines[4].split()[1:7]])
        cell = cellpar_to_cell(cell_dat)
        pbc = [True, True, True]
    else:
        cell = np.zeros((3, 3))
        pbc = [False, False, False]
    symbols = []
    positions = []
    for line in lines[start_line:]:
        if line.startswith('end'):
            break
        flds = line.split()
        symbols.append(flds[7])
        positions.append(flds[1:4])
        atoms.append(Atom(flds[7], flds[1:4]))
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    return atoms