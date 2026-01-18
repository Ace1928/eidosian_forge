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
def replace_group(self, index, func_grp, strategy, bond_order=1, graph_dict=None, strategy_params=None):
    """
        Builds off of Molecule.substitute and MoleculeGraph.substitute_group
        to replace a functional group in self.molecule with a functional group.
        This method also amends self.graph to incorporate the new functional
        group.

        TODO: Figure out how to replace into a ring structure.

        Args:
            index: Index of atom to substitute.
            func_grp: Substituent molecule. There are three options:
                1. Providing an actual molecule as the input. The first atom
                must be a DummySpecies X, indicating the position of
                nearest neighbor. The second atom must be the next
                nearest atom. For example, for a methyl group
                substitution, func_grp should be X-CH3, where X is the
                first site and C is the second site. What the code will
                do is to remove the index site, and connect the nearest
                neighbor to the C atom in CH3. The X-C bond indicates the
                directionality to connect the atoms.
                2. A string name. The molecule will be obtained from the
                relevant template in func_groups.json.
                3. A MoleculeGraph object.
            strategy: Class from pymatgen.analysis.local_env.
            bond_order: A specified bond order to calculate the bond
                length between the attached functional group and the nearest
                neighbor site. Defaults to 1.
            graph_dict: Dictionary representing the bonds of the functional
                group (format: {(u, v): props}, where props is a dictionary of
                properties, including weight. If None, then the algorithm
                will attempt to automatically determine bonds using one of
                a list of strategies defined in pymatgen.analysis.local_env.
            strategy_params: dictionary of keyword arguments for strategy.
                If None, default parameters will be used.
        """
    self.set_node_attributes()
    neighbors = self.get_connected_sites(index)
    if len(neighbors) == 1:
        self.substitute_group(index, func_grp, strategy, bond_order=bond_order, graph_dict=graph_dict, strategy_params=strategy_params)
    else:
        rings = self.find_rings(including=[index])
        if len(rings) != 0:
            raise RuntimeError('Currently functional group replacement cannot occur at an atom within a ring structure.')
        to_remove = set()
        sizes = {}
        disconnected = self.graph.to_undirected()
        disconnected.remove_node(index)
        for neighbor in neighbors:
            sizes[neighbor[2]] = len(nx.descendants(disconnected, neighbor[2]))
        keep = max(sizes, key=lambda x: sizes[x])
        for idx in sizes:
            if idx != keep:
                to_remove.add(idx)
        self.remove_nodes(list(to_remove))
        self.substitute_group(index, func_grp, strategy, bond_order=bond_order, graph_dict=graph_dict, strategy_params=strategy_params)