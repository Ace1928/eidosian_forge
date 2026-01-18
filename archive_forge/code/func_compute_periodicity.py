from __future__ import annotations
import itertools
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from monty.json import MSONable, jsanitize
from networkx.algorithms.components import is_connected
from networkx.algorithms.traversal import bfs_tree
from pymatgen.analysis.chemenv.connectivity.environment_nodes import EnvironmentNode
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.graph_utils import get_delta
from pymatgen.analysis.chemenv.utils.math_utils import get_linearly_independent_vectors
def compute_periodicity(self, algorithm='all_simple_paths') -> None:
    """
        Args:
            algorithm ():
        """
    if algorithm == 'all_simple_paths':
        self.compute_periodicity_all_simple_paths_algorithm()
    elif algorithm == 'cycle_basis':
        self.compute_periodicity_cycle_basis()
    else:
        raise ValueError(f'Algorithm {algorithm!r} is not allowed to compute periodicity')
    self._order_periodicity_vectors()