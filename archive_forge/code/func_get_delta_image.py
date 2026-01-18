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
def get_delta_image(isite1, isite2, data1, data2):
    """
    Helper method to get the delta image between one environment and another
    from the ligand's delta images.
    """
    if data1['start'] == isite1:
        if data2['start'] == isite2:
            return np.array(data1['delta']) - np.array(data2['delta'])
        return np.array(data1['delta']) + np.array(data2['delta'])
    if data2['start'] == isite2:
        return -np.array(data1['delta']) - np.array(data2['delta'])
    return -np.array(data1['delta']) + np.array(data2['delta'])