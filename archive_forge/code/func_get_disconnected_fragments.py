from __future__ import annotations
import copy
import logging
import os.path
import subprocess
import warnings
from collections import defaultdict, namedtuple
from itertools import combinations
from operator import itemgetter
from shutil import which
from typing import TYPE_CHECKING, Any, Callable, cast
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from networkx.drawing.nx_agraph import write_dot
from networkx.readwrite import json_graph
from scipy.spatial import KDTree
from scipy.stats import describe
from pymatgen.core import Lattice, Molecule, PeriodicSite, Structure
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.vis.structure_vtk import EL_COLORS
def get_disconnected_fragments(self, return_index_map: bool=False):
    """Determine if the MoleculeGraph is connected. If it is not, separate the
        MoleculeGraph into different MoleculeGraphs, where each resulting MoleculeGraph is
        a disconnected subgraph of the original. Currently, this function naively assigns
        the charge of the total molecule to a single submolecule. A later effort will be
        to actually accurately assign charge.

        Args:
            return_index_map (bool): If True, return a dictionary that maps the
                new indices to the original indices. Defaults to False.

        NOTE: This function does not modify the original MoleculeGraph. It creates a copy,
        modifies that, and returns two or more new MoleculeGraph objects.

        Returns:
            list[MoleculeGraph]: Each MoleculeGraph is a disconnected subgraph of the original MoleculeGraph.
        """
    if nx.is_weakly_connected(self.graph):
        return [copy.deepcopy(self)]
    original = copy.deepcopy(self)
    sub_mols = []
    new_to_old_index = []
    for c in nx.weakly_connected_components(original.graph):
        subgraph = original.graph.subgraph(c)
        nodes = sorted(subgraph.nodes)
        new_to_old_index += list(nodes)
        mapping = {val: idx for idx, val in enumerate(nodes)}
        charge = self.molecule.charge if 0 in nodes else 0
        new_graph = nx.relabel_nodes(subgraph, mapping)
        species = nx.get_node_attributes(new_graph, 'specie')
        coords = nx.get_node_attributes(new_graph, 'coords')
        raw_props = nx.get_node_attributes(new_graph, 'properties')
        properties: dict[str, Any] = {}
        for prop_set in raw_props.values():
            for prop in prop_set:
                if prop in properties:
                    properties[prop].append(prop_set[prop])
                else:
                    properties[prop] = [prop_set[prop]]
        for k, v in properties.items():
            if len(v) != len(species):
                del properties[k]
        new_mol = Molecule(species, coords, charge=charge, site_properties=properties)
        graph_data = json_graph.adjacency_data(new_graph)
        sub_mols.append(MoleculeGraph(new_mol, graph_data=graph_data))
    if return_index_map:
        return (sub_mols, dict(enumerate(new_to_old_index)))
    return sub_mols