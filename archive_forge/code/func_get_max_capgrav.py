from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
def get_max_capgrav(self, remove=True, insert=True):
    """Give max capacity in mAh/g for inserting and removing a charged ion
        Note that the weight is normalized to the most ion-packed state,
        thus removal of 1 Li from LiFePO4 gives the same capacity as insertion of 1 Li into FePO4.

        Args:
            remove: (bool) whether to allow ion removal
            insert: (bool) whether to allow ion insertion

        Returns:
            max grav capacity in mAh/g
        """
    weight = self.comp.weight
    if insert:
        weight += self.max_ion_insertion * self.working_ion.atomic_mass
    return self._get_max_cap_ah(remove, insert) / (weight / 1000)