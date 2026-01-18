from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def plot_bz(self, path=None, special_points=None, **plotkwargs):
    """Plot the reciprocal cell and default bandpath."""
    bandpath = self.bandpath(path=path, special_points=special_points, npoints=0)
    return bandpath.plot(dimension=self.ndim, **plotkwargs)