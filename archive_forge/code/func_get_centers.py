from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def get_centers(self, scaled=False):
    """Calculate the Wannier centers

        ::

          pos =  L / 2pi * phase(diag(Z))
        """
    coord_wc = np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T / (2 * pi) % 1
    if not scaled:
        coord_wc = np.dot(coord_wc, self.largeunitcell_cc)
    return coord_wc