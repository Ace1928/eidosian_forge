from __future__ import annotations
import collections
import logging
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable, jsanitize
from pymatgen.analysis.chemenv.connectivity.connected_components import ConnectedComponent
from pymatgen.analysis.chemenv.connectivity.environment_nodes import get_environment_node
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
def setup_environments_subgraph(self, environments_symbols):
    raise NotImplementedError