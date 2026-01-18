import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def atoms2kwargs(atoms, use_ase_cell):
    kwargs = {}
    positions = atoms.positions / Bohr
    if use_ase_cell:
        cell = atoms.cell / Bohr
        cell_offset = 0.5 * cell.sum(axis=0)
        positions -= cell_offset
        if atoms.cell.orthorhombic:
            Lsize = 0.5 * np.diag(cell)
            kwargs['lsize'] = [[repr(size) for size in Lsize]]
        else:
            kwargs['latticevectors'] = cell.tolist()
    types = atoms.info.get('types', {})
    coord_block = []
    for sym, pos, tag in zip(atoms.get_chemical_symbols(), positions, atoms.get_tags()):
        if sym == 'X':
            sym = types.get((sym, tag))
            if sym is None:
                raise ValueError('Cannot represent atom X without tags and species info in atoms.info')
        coord_block.append([repr(sym)] + [repr(x) for x in pos])
    kwargs['coordinates'] = coord_block
    npbc = sum(atoms.pbc)
    for c in range(npbc):
        if not atoms.pbc[c]:
            msg = 'Boundary conditions of Atoms object inconsistent with requirements of Octopus.  pbc must be either 000, 100, 110, or 111.'
            raise ValueError(msg)
    kwargs['periodicdimensions'] = npbc
    return kwargs