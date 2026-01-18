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
def elastic_centered_graph(self, start_node=None):
    """
        Args:
            start_node ():

        Returns:
            nx.MultiGraph: Elastic centered subgraph.
        """
    logging.info('In elastic centering')
    n_test_nodes = 0
    start_node = next(iter(self.graph.nodes()))
    n_test_nodes += 1
    centered_connected_subgraph = nx.MultiGraph()
    centered_connected_subgraph.add_nodes_from(self.graph.nodes())
    centered_connected_subgraph.add_edges_from(self.graph.edges(data=True))
    tree = bfs_tree(G=self.graph, source=start_node)
    current_nodes = [start_node]
    nodes_traversed = [start_node]
    inode = 0
    tree_level = 0
    while True:
        tree_level += 1
        logging.debug(f'In tree level {tree_level} ({len(current_nodes)} nodes)')
        new_current_nodes = []
        for node in current_nodes:
            inode += 1
            logging.debug(f'  In node #{inode}/{len(current_nodes)} in level {tree_level} ({node})')
            node_neighbors = list(tree.neighbors(n=node))
            node_edges = centered_connected_subgraph.edges(nbunch=[node], data=True, keys=True)
            for inode_neighbor, node_neighbor in enumerate(node_neighbors):
                logging.debug(f'    Testing neighbor #{inode_neighbor}/{len(node_neighbors)} ({node_neighbor}) of node #{inode} ({node})')
                already_inside = False
                ddeltas = []
                for n1, n2, _key, edata in node_edges:
                    if n1 == node and n2 == node_neighbor or (n2 == node and n1 == node_neighbor):
                        if edata['delta'] == (0, 0, 0):
                            already_inside = True
                            thisdelta = edata['delta']
                        elif edata['start'] == node.isite and edata['end'] != node.isite:
                            thisdelta = edata['delta']
                        elif edata['end'] == node.isite:
                            thisdelta = tuple((-dd for dd in edata['delta']))
                        else:
                            raise ValueError('Should not be here ...')
                        ddeltas.append(thisdelta)
                logging.debug('        ddeltas : ' + ', '.join((f'({', '.join((str(ddd) for ddd in dd))})' for dd in ddeltas)))
                if ddeltas.count((0, 0, 0)) > 1:
                    raise ValueError('Should not have more than one 000 delta ...')
                if already_inside:
                    logging.debug('          Edge inside the cell ... continuing to next neighbor')
                    continue
                logging.debug('          Edge outside the cell ... getting neighbor back inside')
                if (0, 0, 0) in ddeltas:
                    ddeltas.remove((0, 0, 0))
                d_delta = np.array(ddeltas[0], int)
                node_neighbor_edges = centered_connected_subgraph.edges(nbunch=[node_neighbor], data=True, keys=True)
                logging.debug(f'            Delta image from node={node!r} to node_neighbor={node_neighbor!r} : ({', '.join(map(str, d_delta))})')
                for n1, n2, key, edata in node_neighbor_edges:
                    if n1 == node_neighbor and n2 != node_neighbor or (n2 == node_neighbor and n1 != node_neighbor):
                        if edata['start'] == node_neighbor.isite and edata['end'] != node_neighbor.isite:
                            centered_connected_subgraph[n1][n2][key]['delta'] = tuple(np.array(edata['delta'], int) + d_delta)
                        elif edata['end'] == node_neighbor.isite:
                            centered_connected_subgraph[n1][n2][key]['delta'] = tuple(np.array(edata['delta'], int) - d_delta)
                        else:
                            raise ValueError('DUHH')
                        logging.debug(f'                  {n1} to node {n2} now has delta {centered_connected_subgraph[n1][n2][key]['delta']}')
            new_current_nodes.extend(node_neighbors)
            nodes_traversed.extend(node_neighbors)
        current_nodes = new_current_nodes
        if not current_nodes:
            break
    check_centered_connected_subgraph = nx.MultiGraph()
    check_centered_connected_subgraph.add_nodes_from(centered_connected_subgraph.nodes())
    check_centered_connected_subgraph.add_edges_from([edge for edge in centered_connected_subgraph.edges(data=True) if np.allclose(edge[2]['delta'], np.zeros(3))])
    if not is_connected(check_centered_connected_subgraph):
        raise RuntimeError('Could not find a centered graph.')
    return centered_connected_subgraph