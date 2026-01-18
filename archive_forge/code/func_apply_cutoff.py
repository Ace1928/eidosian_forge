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
def apply_cutoff(self, D_N, r_c):
    """Zero elements for interatomic distances larger than the cutoff.

        Parameters:

        D_N: ndarray
            Dynamical/force constant matrix.
        r_c: float
            Cutoff in Angstrom.

        """
    natoms = len(self.indices)
    N = np.prod(self.supercell)
    R_cN = self._lattice_vectors_array
    D_Navav = D_N.reshape((N, natoms, 3, natoms, 3))
    cell_vc = self.atoms.cell.transpose()
    pos_av = self.atoms.get_positions()
    for n in range(N):
        R_v = np.dot(cell_vc, R_cN[:, n])
        posn_av = pos_av + R_v
        for i, a in enumerate(self.indices):
            dist_a = np.sqrt(np.sum((pos_av[a] - posn_av) ** 2, axis=-1))
            i_a = dist_a > r_c
            D_Navav[n, i, :, i_a, :] = 0.0