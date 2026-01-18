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
@classmethod
def from_graph(cls, g) -> Self:
    """
        Constructor for the ConnectedComponent object from a graph of the connected component.

        Args:
            g (MultiGraph): Graph of the connected component.

        Returns:
            ConnectedComponent: The connected component representing the links of a given set of environments.
        """
    return cls(graph=g)