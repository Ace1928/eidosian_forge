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
def find_rings(self, including=None) -> list[list[tuple[int, int]]]:
    """
        Find ring structures in the MoleculeGraph.

        Args:
            including (list[int]): list of site indices. If including is not None, then find_rings
            will only return those rings including the specified sites. By default, this parameter
            is None, and all rings will be returned.

        Returns:
            list[list[tuple[int, int]]]: Each entry will be a ring (cycle, in graph theory terms)
                including the index found in the Molecule. If there is no cycle including an index, the
                value will be an empty list.
        """
    undirected = self.graph.to_undirected()
    directed = undirected.to_directed()
    cycles_nodes = []
    cycles_edges = []
    all_cycles = [sorted(cycle) for cycle in nx.simple_cycles(directed) if len(cycle) > 2]
    unique_sorted = []
    unique_cycles = []
    for cycle in all_cycles:
        if cycle not in unique_sorted:
            unique_sorted.append(cycle)
            unique_cycles.append(cycle)
    if including is None:
        cycles_nodes = unique_cycles
    else:
        for incl in including:
            for cycle in unique_cycles:
                if incl in cycle and cycle not in cycles_nodes:
                    cycles_nodes.append(cycle)
    for cycle in cycles_nodes:
        edges = []
        for idx, itm in enumerate(cycle, start=-1):
            edges.append((cycle[idx], itm))
        cycles_edges.append(edges)
    return cycles_edges