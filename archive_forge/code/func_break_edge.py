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
def break_edge(self, from_index, to_index, allow_reverse=False):
    """
        Remove an edge from the MoleculeGraph.

        Args:
            from_index: int
            to_index: int
            allow_reverse: If allow_reverse is True, then break_edge will
                attempt to break both (from_index, to_index) and, failing that,
                will attempt to break (to_index, from_index).
        """
    existing_edge = self.graph.get_edge_data(from_index, to_index)
    existing_reverse = None
    if existing_edge:
        self.graph.remove_edge(from_index, to_index)
    else:
        if allow_reverse:
            existing_reverse = self.graph.get_edge_data(to_index, from_index)
        if existing_reverse:
            self.graph.remove_edge(to_index, from_index)
        else:
            raise ValueError(f'Edge cannot be broken between {from_index} and {to_index}; no edge exists between those sites.')