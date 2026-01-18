from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_interaction_graph(self, filename=None):
    """
        Get a StructureGraph with edges and weights that correspond to exchange
        interactions and J_ij values, respectively.

        Args:
            filename (str): if not None, save interaction graph to filename.

        Returns:
            igraph (StructureGraph): Exchange interaction graph.
        """
    structure = self.ordered_structures[0]
    sgraph = self.sgraphs[0]
    igraph = StructureGraph.from_empty_graph(structure, edge_weight_name='exchange_constant', edge_weight_units='meV')
    if '<J>' in self.ex_params:
        warning_msg = '\n                Only <J> is available. The interaction graph will not tell\n                you much.\n                '
        logging.warning(warning_msg)
    for i, _node in enumerate(sgraph.graph.nodes):
        connections = sgraph.get_connected_sites(i)
        for c in connections:
            jimage = c[1]
            j = c[2]
            dist = c[-1]
            j_exc = self._get_j_exc(i, j, dist)
            igraph.add_edge(i, j, to_jimage=jimage, weight=j_exc, warn_duplicates=False)
    if filename:
        if not filename.endswith('.json'):
            filename += '.json'
        dumpfn(igraph, filename)
    return igraph