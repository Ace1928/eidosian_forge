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
def alter_edge(self, from_index, to_index, new_weight=None, new_edge_properties=None):
    """
        Alters either the weight or the edge_properties of
        an edge in the MoleculeGraph.

        Args:
            from_index: int
            to_index: int
            new_weight: alter_edge does not require
                that weight be altered. As such, by default, this
                is None. If weight is to be changed, it should be a
                float.
            new_edge_properties: alter_edge does not require
                that edge_properties be altered. As such, by default,
                this is None. If any edge properties are to be changed,
                it should be a dictionary of edge properties to be changed.
        """
    existing_edge = self.graph.get_edge_data(from_index, to_index)
    if not existing_edge:
        raise ValueError(f'Edge between {from_index} and {to_index} cannot be altered; no edge exists between those sites.')
    if new_weight is not None:
        self.graph[from_index][to_index][0]['weight'] = new_weight
    if new_edge_properties is not None:
        for prop in new_edge_properties:
            self.graph[from_index][to_index][0][prop] = new_edge_properties[prop]