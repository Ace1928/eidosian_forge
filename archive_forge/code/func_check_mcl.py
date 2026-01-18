from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def check_mcl(a, b, c, alpha):
    if not (b <= c and alpha < 90):
        raise UnconventionalLattice('Expected b <= c, alpha < 90; got a={}, b={}, c={}, alpha={}'.format(a, b, c, alpha))