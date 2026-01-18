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
def split_molecule_subgraphs(self, bonds, allow_reverse=False, alterations=None):
    """
        Split MoleculeGraph into two or more MoleculeGraphs by
        breaking a set of bonds. This function uses
        MoleculeGraph.break_edge repeatedly to create
        disjoint graphs (two or more separate molecules).
        This function does not only alter the graph
        information, but also changes the underlying
        Molecules.
        If the bonds parameter does not include sufficient
        bonds to separate two molecule fragments, then this
        function will fail.
        Currently, this function naively assigns the charge
        of the total molecule to a single submolecule. A
        later effort will be to actually accurately assign
        charge.
        NOTE: This function does not modify the original
        MoleculeGraph. It creates a copy, modifies that, and
        returns two or more new MoleculeGraph objects.

        Args:
            bonds: list of tuples (from_index, to_index)
                representing bonds to be broken to split the MoleculeGraph.
            alterations: a dict {(from_index, to_index): alt},
                where alt is a dictionary including weight and/or edge
                properties to be changed following the split.
            allow_reverse: If allow_reverse is True, then break_edge will
                attempt to break both (from_index, to_index) and, failing that,
                will attempt to break (to_index, from_index).

        Returns:
            list of MoleculeGraphs.
        """
    self.set_node_attributes()
    original = copy.deepcopy(self)
    for bond in bonds:
        original.break_edge(bond[0], bond[1], allow_reverse=allow_reverse)
    if nx.is_weakly_connected(original.graph):
        raise MolGraphSplitError('Cannot split molecule; MoleculeGraph is still connected.')
    if alterations is not None:
        for u, v in alterations:
            if 'weight' in alterations[u, v]:
                weight = alterations[u, v].pop('weight')
                edge_properties = alterations[u, v] if len(alterations[u, v]) != 0 else None
                original.alter_edge(u, v, new_weight=weight, new_edge_properties=edge_properties)
            else:
                original.alter_edge(u, v, new_edge_properties=alterations[u, v])
    return original.get_disconnected_fragments()