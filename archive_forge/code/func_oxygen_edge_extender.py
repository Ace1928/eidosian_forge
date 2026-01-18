from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def oxygen_edge_extender(mol_graph: MoleculeGraph) -> MoleculeGraph:
    """
    Identify and add missed O-C or O-H bonds. This is particularly
    important when oxygen is forming three bonds, e.g. in H3O+ or XOH2+.
    See https://github.com/materialsproject/pymatgen/pull/2903 for details.

    Args:
        mol_graph (MoleculeGraph): molecule graph to extend

    Returns:
        MoleculeGraph: object with additional O-C or O-H bonds added (if any found)
    """
    num_new_edges = 0
    for idx in mol_graph.graph.nodes():
        if mol_graph.graph.nodes()[idx]['specie'] == 'O':
            neighbors = [site[2] for site in mol_graph.get_connected_sites(idx)]
            for ii, site in enumerate(mol_graph.molecule):
                is_O_C_bond = str(site.specie) == 'C' and site.distance(mol_graph.molecule[idx]) < 1.5
                is_O_H_bond = str(site.specie) == 'H' and site.distance(mol_graph.molecule[idx]) < 1
                if ii != idx and ii not in neighbors and (is_O_C_bond or is_O_H_bond):
                    mol_graph.add_edge(idx, ii)
                    num_new_edges += 1
    return mol_graph