from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def get_symmetrically_distinct_miller_indices(structure: Structure, max_index: int, return_hkil: bool=False) -> list:
    """Find all symmetrically distinct indices below a certain max-index
    for a given structure. Analysis is based on the symmetry of the
    reciprocal lattice of the structure.

    Args:
        structure (Structure): The input structure.
        max_index (int): The maximum index. For example, 1 means that
            (100), (110), and (111) are returned for the cubic structure.
            All other indices are equivalent to one of these.
        return_hkil (bool): Whether to return hkil (True) form of Miller
            index for hexagonal systems, or hkl (False).
    """
    rng = list(range(-max_index, max_index + 1))[::-1]
    conv_hkl_list = [miller for miller in itertools.product(rng, rng, rng) if any((i != 0 for i in miller))]
    conv_hkl_list = sorted(conv_hkl_list, key=lambda x: max(np.abs(x)))
    spg_analyzer = SpacegroupAnalyzer(structure)
    if spg_analyzer.get_crystal_system() == 'trigonal':
        transf = spg_analyzer.get_conventional_to_primitive_transformation_matrix()
        miller_list: list[tuple[int, int, int]] = [hkl_transformation(transf, hkl) for hkl in conv_hkl_list]
        prim_structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
        symm_ops = prim_structure.lattice.get_recp_symmetry_operation()
    else:
        miller_list = conv_hkl_list
        symm_ops = structure.lattice.get_recp_symmetry_operation()
    unique_millers: list = []
    unique_millers_conv: list = []
    for idx, miller in enumerate(miller_list):
        denom = abs(reduce(gcd, miller))
        miller = cast(tuple[int, int, int], tuple((int(idx / denom) for idx in miller)))
        if not _is_in_miller_family(miller, unique_millers, symm_ops):
            if spg_analyzer.get_crystal_system() == 'trigonal':
                unique_millers.append(miller)
                denom = abs(reduce(gcd, conv_hkl_list[idx]))
                cmiller = tuple((int(idx / denom) for idx in conv_hkl_list[idx]))
                unique_millers_conv.append(cmiller)
            else:
                unique_millers.append(miller)
                unique_millers_conv.append(miller)
    if return_hkil and spg_analyzer.get_crystal_system() in {'trigonal', 'hexagonal'}:
        return [(hkl[0], hkl[1], -1 * hkl[0] - hkl[1], hkl[2]) for hkl in unique_millers_conv]
    return unique_millers_conv