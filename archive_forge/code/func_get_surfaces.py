import math
import numpy as np
from ase import Atoms
from ase.cluster.base import ClusterBase
def get_surfaces(self):
    """Returns the miller indexs of the stored surfaces of the cluster."""
    if self.surfaces is not None:
        return self.surfaces.copy()
    else:
        return None