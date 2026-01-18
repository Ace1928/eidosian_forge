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
def build_unique_fragments(self):
    """
        Find all possible fragment combinations of the MoleculeGraphs (in other
        words, all connected induced subgraphs).
        """
    self.set_node_attributes()
    graph = self.graph.to_undirected()
    frag_dict = {}
    for ii in range(1, len(self.molecule)):
        for combination in combinations(graph.nodes, ii):
            comp = []
            for idx in combination:
                comp.append(str(self.molecule[idx].specie))
            comp = ''.join(sorted(comp))
            subgraph = nx.subgraph(graph, combination)
            if nx.is_connected(subgraph):
                key = f'{comp} {len(subgraph.edges())}'
                if key not in frag_dict:
                    frag_dict[key] = [copy.deepcopy(subgraph)]
                else:
                    frag_dict[key].append(copy.deepcopy(subgraph))
    unique_frag_dict = {}
    for key, fragments in frag_dict.items():
        unique_frags = []
        for frag in fragments:
            found = False
            for fragment in unique_frags:
                if _isomorphic(frag, fragment):
                    found = True
                    break
            if not found:
                unique_frags.append(frag)
        unique_frag_dict[key] = copy.deepcopy(unique_frags)
    unique_mol_graph_dict = {}
    for key, fragments in unique_frag_dict.items():
        unique_mol_graph_list = []
        for fragment in fragments:
            mapping = {edge: idx for idx, edge in enumerate(sorted(fragment.nodes))}
            remapped = nx.relabel_nodes(fragment, mapping)
            species = nx.get_node_attributes(remapped, 'specie')
            coords = nx.get_node_attributes(remapped, 'coords')
            edges = {}
            for from_index, to_index, key in remapped.edges:
                edge_props = fragment.get_edge_data(from_index, to_index, key=key)
                edges[from_index, to_index] = edge_props
            unique_mol_graph_list.append(self.from_edges(Molecule(species=species, coords=coords, charge=self.molecule.charge), edges))
        alph_formula = unique_mol_graph_list[0].molecule.composition.alphabetical_formula
        frag_key = f'{alph_formula} E{len(unique_mol_graph_list[0].graph.edges())}'
        unique_mol_graph_dict[frag_key] = copy.deepcopy(unique_mol_graph_list)
    return unique_mol_graph_dict