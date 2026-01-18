import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def idcdchunks(fd, ref_atoms, aligned):
    """Yield unprocessed chunks for each image."""
    if ref_atoms:
        symbols = ref_atoms.get_chemical_symbols()
    else:
        symbols = None
    dtype, natoms, nsteps, header_end = _read_metainfo(fd)
    bytes_per_step = _bytes_per_timestep(natoms)
    fd.seek(header_end)
    for i in range(nsteps):
        fd.seek(bytes_per_step * i + header_end)
        yield DCDChunk(fd, dtype, natoms, symbols, aligned)