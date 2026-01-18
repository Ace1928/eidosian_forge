from math import pi, sqrt
import warnings
from pathlib import Path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import ase
import ase.units as units
from ase.parallel import world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils.filecache import MultiFileJSONCache
@supercell.setter
def supercell(self, supercell):
    assert len(supercell) == 3
    self._supercell = tuple(supercell)
    self.define_offset()
    self._lattice_vectors_array = self.compute_lattice_vectors()