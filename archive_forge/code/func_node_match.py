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
def node_match(n1, n2):
    return n1['specie'] == n2['specie']