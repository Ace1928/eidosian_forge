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
def coordination_sequence(self, source_node, path_size=5, coordination='number', include_source=False):
    """Get the coordination sequence for a given node.

        Args:
            source_node: Node for which the coordination sequence is computed.
            path_size: Maximum length of the path for the coordination sequence.
            coordination: Type of coordination sequence. The default ("number") corresponds to the number
                of environment nodes that are reachable by following paths of sizes between 1 and path_size.
                For coordination "env:number", this resulting coordination sequence is a sequence of dictionaries
                mapping the type of environment to the number of such environment reachable by following paths of
                sizes between 1 and path_size.
            include_source: Whether to include the source_node in the coordination sequence.

        Returns:
            dict: Mapping between the nth "layer" of the connected component with the corresponding coordination.

        Examples:
            The corner-sharing octahedral framework (as in perovskites) have the following coordination sequence (up to
            a path of size 6) :
            {1: 6, 2: 18, 3: 38, 4: 66, 5: 102, 6: 146}
            Considering both the octahedrons and the cuboctahedrons of the typical BaTiO3 perovskite, the "env:number"
            coordination sequence (up to a path of size 6) starting on the Ti octahedron and Ba cuboctahedron
            are the following :
            Starting on the Ti octahedron : {1: {'O:6': 6, 'C:12': 8}, 2: {'O:6': 26, 'C:12': 48},
                                             3: {'O:6': 90, 'C:12': 128}, 4: {'O:6': 194, 'C:12': 248},
                                             5: {'O:6': 338, 'C:12': 408}, 6: {'O:6': 522, 'C:12': 608}}
            Starting on the Ba cuboctahedron : {1: {'O:6': 8, 'C:12': 18}, 2: {'O:6': 48, 'C:12': 74},
                                                3: {'O:6': 128, 'C:12': 170}, 4: {'O:6': 248, 'C:12': 306},
                                                5: {'O:6': 408, 'C:12': 482}, 6: {'O:6': 608, 'C:12': 698}}
            If include_source is set to True, the source node is included in the sequence, e.g. for the corner-sharing
            octahedral framework : {0: 1, 1: 6, 2: 18, 3: 38, 4: 66, 5: 102, 6: 146}. For the "env:number" coordination
            starting on a Ba cuboctahedron (as shown above), the coordination sequence is then :
            {0: {'C:12': 1}, 1: {'O:6': 8, 'C:12': 18}, 2: {'O:6': 48, 'C:12': 74}, 3: {'O:6': 128, 'C:12': 170},
             4: {'O:6': 248, 'C:12': 306}, 5: {'O:6': 408, 'C:12': 482}, 6: {'O:6': 608, 'C:12': 698}}
        """
    if source_node not in self._connected_subgraph:
        raise ValueError('Node not in Connected Component. Cannot find coordination sequence.')
    current_delta = (0, 0, 0)
    current_ends = [(source_node, current_delta)]
    visited = {(source_node.isite, *current_delta)}
    path_len = 0
    cseq = {}
    if include_source:
        if coordination == 'number':
            cseq[0] = 1
        elif coordination == 'env:number':
            cseq[0] = {source_node.coordination_environment: 1}
        else:
            raise ValueError(f'Coordination type {coordination!r} is not valid for coordination_sequence.')
    while path_len < path_size:
        new_ends = []
        for current_node_end, current_delta_end in current_ends:
            for nb in self._connected_subgraph.neighbors(current_node_end):
                for edata in self._connected_subgraph[current_node_end][nb].values():
                    new_delta = current_delta_end + get_delta(current_node_end, nb, edata)
                    if (nb.isite, *new_delta) not in visited:
                        new_ends.append((nb, new_delta))
                        visited.add((nb.isite, *new_delta))
                    if nb.isite == current_node_end.isite:
                        new_delta = current_delta_end - get_delta(current_node_end, nb, edata)
                        if (nb.isite, *new_delta) not in visited:
                            new_ends.append((nb, new_delta))
                            visited.add((nb.isite, *new_delta))
        current_ends = new_ends
        path_len += 1
        if coordination == 'number':
            cseq[path_len] = len(current_ends)
        elif coordination == 'env:number':
            envs = [end.coordination_environment for end, _ in current_ends]
            cseq[path_len] = {env: envs.count(env) for env in set(envs)}
        else:
            raise ValueError(f'Coordination type {coordination!r} is not valid for coordination_sequence.')
    return cseq