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
def get_dimensionality_larsen(bonded_structure):
    """
    Gets the dimensionality of a bonded structure.

    The dimensionality of the structure is the highest dimensionality of all
    structure components. This method is very robust and can handle
    many tricky structures, regardless of structure type or improper connections
    due to periodic boundary conditions.

    Requires a StructureGraph object as input. This can be generated using one
    of the NearNeighbor classes. For example, using the CrystalNN class:

        bonded_structure = CrystalNN().get_bonded_structure(structure)

    Based on the modified breadth-first-search algorithm described in:

    P. M. Larsen, M. Pandey, M. Strange, K. W. Jacobsen. Definition of a
    scoring parameter to identify low-dimensional materials components.
    Phys. Rev. Materials 3, 034003 (2019).

    Args:
        bonded_structure (StructureGraph): A structure with bonds, represented
            as a pymatgen structure graph. For example, generated using the
            CrystalNN.get_bonded_structure() method.

    Returns:
        int: The dimensionality of the structure.
    """
    return max((c['dimensionality'] for c in get_structure_components(bonded_structure)))