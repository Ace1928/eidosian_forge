from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
@staticmethod
def parse_struct(path_dir):
    """Parses boltztrap.struct file (only the volume).

        Args:
            path_dir: (str) dir containing the boltztrap.struct file

        Returns:
            float: volume of the structure in Angstrom^3
        """
    with open(f'{path_dir}/boltztrap.struct') as file:
        tokens = file.readlines()
        return Lattice([[Length(float(tokens[i].split()[j]), 'bohr').to('ang') for j in range(3)] for i in range(1, 4)]).volume