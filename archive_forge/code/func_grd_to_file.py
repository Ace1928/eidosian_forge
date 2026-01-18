import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def grd_to_file(atoms, grd_file, new_file):
    """ Reads grd_file and converts data to cube format and writes to
    cube_file.

    Note: content of grd_file and atoms object are assumed to match with the
          same orientation.

    Parameters
    -----------
    atoms (Atoms object): atoms object grd_file data is for
    grd_file (str): filename of .grd file
    new_file (str): filename to write grd-data to, must be ASE format
                    that supports data argument
    """
    from ase.io import write
    atoms_copy = atoms.copy()
    data, cell, origin = read_grd(grd_file)
    atoms_copy.cell = cell
    atoms_copy.positions += origin
    write(new_file, atoms_copy, data=data)