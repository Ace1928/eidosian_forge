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
def find_connected_atoms(struct, tolerance=0.45, ldict=None):
    """
    Finds bonded atoms and returns a adjacency matrix of bonded atoms.

    Author: "Gowoon Cheon"
    Email: "gcheon@stanford.edu"

    Args:
        struct (Structure): Input structure
        tolerance: length in angstroms used in finding bonded atoms. Two atoms
            are considered bonded if (radius of atom 1) + (radius of atom 2) +
            (tolerance) < (distance between atoms 1 and 2). Default
            value = 0.45, the value used by JMol and Cheon et al.
        ldict: dictionary of bond lengths used in finding bonded atoms. Values
            from JMol are used as default

    Returns:
        np.ndarray: A numpy array of shape (number of atoms, number of atoms);
            If any image of atom j is bonded to atom i with periodic boundary
            conditions, the matrix element [atom i, atom j] is 1.
    """
    if ldict is None:
        ldict = JmolNN().el_radius
    n_atoms = len(struct.species)
    fc = np.array(struct.frac_coords)
    fc_copy = np.repeat(fc[:, :, np.newaxis], 27, axis=2)
    neighbors = np.array(list(itertools.product([0, 1, -1], [0, 1, -1], [0, 1, -1]))).T
    neighbors = np.repeat(neighbors[np.newaxis, :, :], 1, axis=0)
    fc_diff = fc_copy - neighbors
    species = list(map(str, struct.species))
    for ii, item in enumerate(species):
        if item not in ldict:
            species[ii] = str(Species.from_str(item).element)
    connected_matrix = np.zeros((n_atoms, n_atoms))
    for ii in range(n_atoms):
        for jj in range(ii + 1, n_atoms):
            max_bond_length = ldict[species[ii]] + ldict[species[jj]] + tolerance
            frac_diff = fc_diff[jj] - fc_copy[ii]
            distance_ij = np.dot(struct.lattice.matrix.T, frac_diff)
            if sum(np.linalg.norm(distance_ij, axis=0) < max_bond_length) > 0:
                connected_matrix[ii, jj] = 1
                connected_matrix[jj, ii] = 1
    return connected_matrix