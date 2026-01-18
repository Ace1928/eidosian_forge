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
def get_dimensionality_cheon(structure_raw, tolerance=0.45, ldict=None, standardize=True, larger_cell=False):
    """
    Algorithm for finding the dimensions of connected subunits in a structure.
    This method finds the dimensionality of the material even when the material
    is not layered along low-index planes, or does not have flat
    layers/molecular wires.

    Author: "Gowoon Cheon"
    Email: "gcheon@stanford.edu"

    See details at :

    Cheon, G.; Duerloo, K.-A. N.; Sendek, A. D.; Porter, C.; Chen, Y.; Reed,
    E. J. Data Mining for New Two- and One-Dimensional Weakly Bonded Solids and
    Lattice-Commensurate Heterostructures. Nano Lett. 2017.

    Args:
        structure_raw (Structure): A pymatgen Structure object.
        tolerance (float): length in angstroms used in finding bonded atoms.
            Two atoms are considered bonded if (radius of atom 1) + (radius of
            atom 2) + (tolerance) < (distance between atoms 1 and 2). Default
            value = 0.45, the value used by JMol and Cheon et al.
        ldict (dict): dictionary of bond lengths used in finding bonded atoms.
            Values from JMol are used as default
        standardize: works with conventional standard structures if True. It is
            recommended to keep this as True.
        larger_cell: tests with 3x3x3 supercell instead of 2x2x2. Testing with
            2x2x2 supercell is faster but misclassifies rare interpenetrated 3D
             structures. Testing with a larger cell circumvents this problem

    Returns:
        str: dimension of the largest cluster as a string. If there are ions
            or molecules it returns 'intercalated ion/molecule'
    """
    if ldict is None:
        ldict = JmolNN().el_radius
    if standardize:
        structure = SpacegroupAnalyzer(structure_raw).get_conventional_standard_structure()
    else:
        structure = structure_raw
    structure_save = copy.copy(structure_raw)
    connected_list1 = find_connected_atoms(structure, tolerance=tolerance, ldict=ldict)
    max1, min1, _clusters1 = find_clusters(structure, connected_list1)
    if larger_cell:
        structure.make_supercell(np.eye(3) * 3)
        connected_list3 = find_connected_atoms(structure, tolerance=tolerance, ldict=ldict)
        max3, min3, _clusters3 = find_clusters(structure, connected_list3)
        if min3 == min1:
            dim = '0D' if max3 == max1 else 'intercalated molecule'
        else:
            dim = np.log2(float(max3) / max1) / np.log2(3)
            if dim == int(dim):
                dim = str(int(dim)) + 'D'
            else:
                return None
    else:
        structure.make_supercell(np.eye(3) * 2)
        connected_list2 = find_connected_atoms(structure, tolerance=tolerance, ldict=ldict)
        max2, min2, _clusters2 = find_clusters(structure, connected_list2)
        if min2 == 1:
            dim = 'intercalated ion'
        elif min2 == min1:
            dim = '0D' if max2 == max1 else 'intercalated molecule'
        else:
            dim = np.log2(float(max2) / max1)
            if dim == int(dim):
                dim = str(int(dim)) + 'D'
            else:
                structure = copy.copy(structure_save)
                structure.make_supercell(np.eye(3) * 3)
                connected_list3 = find_connected_atoms(structure, tolerance=tolerance, ldict=ldict)
                max3, min3, _clusters3 = find_clusters(structure, connected_list3)
                if min3 == min2:
                    dim = '0D' if max3 == max2 else 'intercalated molecule'
                else:
                    dim = np.log2(float(max3) / max1) / np.log2(3)
                    if dim == int(dim):
                        dim = str(int(dim)) + 'D'
                    else:
                        return None
    return dim