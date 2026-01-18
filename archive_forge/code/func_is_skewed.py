import numpy as np
from ase.geometry import wrap_positions
def is_skewed(self):
    """Test if a lammps cell is not tetragonal

        :returns: bool
        :rtype: bool

        """
    cell_sq = self.lammps_cell ** 2
    return np.sum(np.tril(cell_sq, -1)) / np.sum(np.diag(cell_sq)) > self.tolerance