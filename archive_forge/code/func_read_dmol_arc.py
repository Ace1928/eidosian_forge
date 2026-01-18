from datetime import datetime
import numpy as np
from ase import Atom, Atoms
from ase.geometry.cell import cell_to_cellpar, cellpar_to_cell
from ase.units import Bohr
from ase.utils import reader, writer
@reader
def read_dmol_arc(fd, index=-1):
    """ Read a dmol arc-file and return a series of Atoms objects (images). """
    lines = fd.readlines()
    images = []
    if lines[1].startswith('PBC=ON'):
        pbc = True
    elif lines[1].startswith('PBC=OFF'):
        pbc = False
    else:
        raise RuntimeError('Could not read pbc from second line in file')
    i = 0
    while i < len(lines):
        cell = np.zeros((3, 3))
        symbols = []
        positions = []
        if lines[i].startswith('!DATE'):
            if pbc:
                cell_dat = np.array([float(fld) for fld in lines[i + 1].split()[1:7]])
                cell = cellpar_to_cell(cell_dat)
                i += 1
            i += 1
            while not lines[i].startswith('end'):
                flds = lines[i].split()
                symbols.append(flds[7])
                positions.append(flds[1:4])
                i += 1
            image = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
            images.append(image)
        if len(images) == index:
            return images[-1]
        i += 1
    if isinstance(index, int):
        return images[index]
    else:
        from ase.io.formats import index2range
        indices = index2range(index, len(images))
        return [images[j] for j in indices]