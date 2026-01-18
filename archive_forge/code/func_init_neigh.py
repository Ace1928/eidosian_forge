import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
def init_neigh(self, neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh):
    """Initialize neighbor list, either an ASE-native neighborlist
        or one created using the neighlist module in kimpy
        """
    neigh_list_object_type = neighborlist.ASENeighborList if self.ase_neigh else neighborlist.KimpyNeighborList
    self.neigh = neigh_list_object_type(self.compute_args, neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh, self.debug)