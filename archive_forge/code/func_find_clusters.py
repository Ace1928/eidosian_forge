from __future__ import annotations
import copy
import itertools
from collections import defaultdict
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_analyzer import get_max_bond_lengths
from pymatgen.core import Molecule, Species, Structure
from pymatgen.core.lattice import get_integer_index
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def find_clusters(struct, connected_matrix):
    """
    Finds bonded clusters of atoms in the structure with periodic boundary
    conditions.

    If there are atoms that are not bonded to anything, returns [0,1,0]. (For
    faster computation time)

    Author: Gowoon Cheon
    Email: gcheon@stanford.edu

    Args:
        struct (Structure): Input structure
        connected_matrix: Must be made from the same structure with
            find_connected_atoms() function.

    Returns:
        max_cluster: the size of the largest cluster in the crystal structure
        min_cluster: the size of the smallest cluster in the crystal structure
        clusters: list of bonded clusters found here, clusters are formatted as
        sets of indices of atoms
    """
    n_atoms = len(struct.species)
    if n_atoms == 0:
        return [0, 0, 0]
    if 0 in np.sum(connected_matrix, axis=0):
        return [0, 1, 0]
    cluster_sizes = []
    clusters = []
    visited = [False for item in range(n_atoms)]
    connected_matrix += np.eye(len(connected_matrix))

    def visit(atom, atom_cluster):
        visited[atom] = True
        new_cluster = set(np.where(connected_matrix[atom] != 0)[0]) | atom_cluster
        atom_cluster = new_cluster
        for new_atom in atom_cluster:
            if not visited[new_atom]:
                visited[new_atom] = True
                atom_cluster = visit(new_atom, atom_cluster)
        return atom_cluster
    for idx in range(n_atoms):
        if not visited[idx]:
            atom_cluster = set()
            cluster = visit(idx, atom_cluster)
            clusters.append(cluster)
            cluster_sizes.append(len(cluster))
    max_cluster = max(cluster_sizes)
    min_cluster = min(cluster_sizes)
    return [max_cluster, min_cluster, clusters]