from __future__ import annotations
import os
from pymatgen.io.vasp import Potcar
def gen_potcar(dirname, filename):
    """Generate POTCAR from POTCAR.spec in directories.

    Args:
        dirname (str): Directory name.
        filename (str): Filename in directory.
    """
    if filename == 'POTCAR.spec':
        fullpath = os.path.join(dirname, filename)
        with open(fullpath) as file:
            elements = file.readlines()
        symbols = [el.strip() for el in elements if el.strip() != '']
        potcar = Potcar(symbols)
        potcar.write_file(f'{dirname}/POTCAR')