from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def get_nn_info(self, structure: Structure, n, use_weights: bool=False):
    """
        Get coordination number, CN, of site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).
                True is not implemented for LobsterNeighbors

        Returns:
            cn (integer or float): coordination number.
        """
    if use_weights:
        raise ValueError('LobsterEnv cannot use weights')
    if len(structure) != len(self.structure):
        raise ValueError('The wrong structure was provided')
    return self.sg_list[n]