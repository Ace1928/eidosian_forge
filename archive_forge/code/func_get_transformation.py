from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def get_transformation(self, cell, eps=1e-08):
    T = cell.dot(np.linalg.pinv(self.tocell()))
    msg = 'This transformation changes the length/area/volume of the cell'
    assert np.isclose(np.abs(np.linalg.det(T[:self.ndim, :self.ndim])), 1, atol=eps), msg
    return T