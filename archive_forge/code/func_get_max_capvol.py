from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
def get_max_capvol(self, remove=True, insert=True, volume=None):
    """Give max capacity in mAh/cc for inserting and removing a charged ion into base structure.

        Args:
            remove: (bool) whether to allow ion removal
            insert: (bool) whether to allow ion insertion
            volume: (float) volume to use for normalization (default=volume of initial structure)

        Returns:
            max vol capacity in mAh/cc
        """
    vol = volume or self.struct_oxid.volume
    return self._get_max_cap_ah(remove, insert) * 1000 * 1e+24 / (vol * const.N_A)