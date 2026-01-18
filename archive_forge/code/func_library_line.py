from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def library_line(file_name):
    """Specifies GULP library file to read species and potential parameters.
        If using library don't specify species and potential
        in the input file and vice versa. Make sure the elements of
        structure are in the library file.

        Args:
            file_name: Name of GULP library file

        Returns:
            GULP input string specifying library option
        """
    gulp_lib_set = 'GULP_LIB' in os.environ

    def readable(file):
        return os.path.isfile(file) and os.access(file, os.R_OK)
    gin = ''
    dirpath, _fname = os.path.split(file_name)
    if dirpath and readable(file_name):
        gin = f'library {file_name}'
    else:
        fpath = os.path.join(os.getcwd(), file_name)
        if readable(fpath):
            gin = f'library {fpath}'
        elif gulp_lib_set:
            fpath = os.path.join(os.environ['GULP_LIB'], file_name)
            if readable(fpath):
                gin = f'library {file_name}'
    if gin:
        return gin + '\n'
    raise GulpError('GULP library not found')