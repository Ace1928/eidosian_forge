import os
from warnings import warn
from glob import glob
import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.utils import writer
def read_eon(fileobj, index=-1):
    """Reads an EON reactant.con file.  If *fileobj* is the name of a
    "states" directory created by EON, all the structures will be read."""
    if isinstance(fileobj, str):
        if os.path.isdir(fileobj):
            return read_states(fileobj)
        else:
            fd = open(fileobj)
    else:
        fd = fileobj
    more_images_to_read = True
    images = []
    first_line = fd.readline()
    while more_images_to_read:
        comment = first_line.strip()
        fd.readline()
        cell_lengths = fd.readline().split()
        cell_angles = fd.readline().split()
        cell_angles = [cell_angles[2], cell_angles[1], cell_angles[0]]
        cellpar = [float(x) for x in cell_lengths + cell_angles]
        fd.readline()
        fd.readline()
        ntypes = int(fd.readline())
        natoms = [int(n) for n in fd.readline().split()]
        atommasses = [float(m) for m in fd.readline().split()]
        symbols = []
        coords = []
        masses = []
        fixed = []
        for n in range(ntypes):
            symbol = fd.readline().strip()
            symbols.extend([symbol] * natoms[n])
            masses.extend([atommasses[n]] * natoms[n])
            fd.readline()
            for i in range(natoms[n]):
                row = fd.readline().split()
                coords.append([float(x) for x in row[:3]])
                fixed.append(bool(int(row[3])))
        atoms = Atoms(symbols=symbols, positions=coords, masses=masses, cell=cellpar_to_cell(cellpar), constraint=FixAtoms(mask=fixed), info=dict(comment=comment))
        images.append(atoms)
        first_line = fd.readline()
        if first_line == '':
            more_images_to_read = False
    if isinstance(fileobj, str):
        fd.close()
    if not index:
        return images
    else:
        return images[index]