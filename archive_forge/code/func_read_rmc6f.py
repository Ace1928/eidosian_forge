import re
import time
import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.cell import Cell
@reader
def read_rmc6f(filename, atom_type_map=None):
    """
    Parse a RMCProfile rmc6f file into ASE Atoms object

    Parameters
    ----------
    filename : file|str
        A file like object or filename.
    atom_type_map: dict{str:str}
        Map of atom types for conversions. Mainly used if there is
        an atom type in the file that is not supported by ASE but
        want to map to a supported atom type instead.

        Example to map deuterium to hydrogen:
        atom_type_map = { 'D': 'H' }

    Returns
    ------
    structure : Atoms
        The Atoms object read in from the rmc6f file.
    """
    fd = filename
    lines = fd.readlines()
    pos, cell = _read_process_rmc6f_lines_to_pos_and_cell(lines)
    if atom_type_map is None:
        symbols = [atom[0] for atom in pos.values()]
        atom_type_map = {atype: atype for atype in symbols}
    for atom in pos.values():
        atom[0] = atom_type_map[atom[0]]
    symbols = []
    scaled_positions = []
    spin = None
    magmoms = []
    for atom in pos.values():
        if len(atom) == 4:
            element, x, y, z = atom
        else:
            element, x, y, z, spin = atom
        element = atom_type_map[element]
        symbols.append(element)
        scaled_positions.append([x, y, z])
        if spin is not None:
            magmoms.append(spin)
    atoms = Atoms(scaled_positions=scaled_positions, symbols=symbols, cell=cell, magmoms=magmoms, pbc=[True, True, True])
    return atoms