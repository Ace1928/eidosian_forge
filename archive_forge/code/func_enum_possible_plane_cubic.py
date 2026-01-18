from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def enum_possible_plane_cubic(plane_cutoff, r_axis, r_angle):
    """
        Find all possible plane combinations for GBs given a rotation axis and angle for
        cubic system, and classify them to different categories, including 'Twist',
        'Symmetric tilt', 'Normal tilt', 'Mixed' GBs.

        Args:
            plane_cutoff (int): the cutoff of plane miller index.
            r_axis (list of 3 integers, e.g. u, v, w):
                the rotation axis of the grain boundary, with the format of [u,v,w].
            r_angle (float): rotation angle of the GBs.

        Returns:
            dict: all combinations with keys as GB type, e.g. 'Twist','Symmetric tilt',etc.
                and values as the combination of the two plane miller index (GB plane and joining plane).
        """
    all_combinations = {}
    all_combinations['Symmetric tilt'] = []
    all_combinations['Twist'] = []
    all_combinations['Normal tilt'] = []
    all_combinations['Mixed'] = []
    sym_plane = symm_group_cubic([[1, 0, 0], [1, 1, 0]])
    j = np.arange(0, plane_cutoff + 1)
    combination = []
    for idx in product(j, repeat=3):
        if sum(abs(np.array(idx))) != 0:
            combination.append(list(idx))
        if len(np.nonzero(idx)[0]) == 3:
            for i1 in range(3):
                new_i = list(idx).copy()
                new_i[i1] = -1 * new_i[i1]
                combination.append(new_i)
        elif len(np.nonzero(idx)[0]) == 2:
            new_i = list(idx).copy()
            new_i[np.nonzero(idx)[0][0]] = -1 * new_i[np.nonzero(idx)[0][0]]
            combination.append(new_i)
    miller = np.array(combination)
    miller = miller[np.argsort(np.linalg.norm(miller, axis=1))]
    for val in miller:
        if reduce(gcd, val) == 1:
            matrix = GrainBoundaryGenerator.get_trans_mat(r_axis, r_angle, surface=val, quick_gen=True)
            vec = np.cross(matrix[1][0], matrix[1][1])
            miller2 = GrainBoundaryGenerator.vec_to_surface(vec)
            if np.all(np.abs(np.array(miller2)) <= plane_cutoff):
                cos_1 = abs(np.dot(val, r_axis) / np.linalg.norm(val) / np.linalg.norm(r_axis))
                if 1 - cos_1 < 1e-05:
                    all_combinations['Twist'].append([list(val), miller2])
                elif cos_1 < 1e-08:
                    sym_tilt = False
                    if np.sum(np.abs(val)) == np.sum(np.abs(miller2)):
                        ave = (np.array(val) + np.array(miller2)) / 2
                        ave1 = (np.array(val) - np.array(miller2)) / 2
                        for plane in sym_plane:
                            cos_2 = abs(np.dot(ave, plane) / np.linalg.norm(ave) / np.linalg.norm(plane))
                            cos_3 = abs(np.dot(ave1, plane) / np.linalg.norm(ave1) / np.linalg.norm(plane))
                            if 1 - cos_2 < 1e-05 or 1 - cos_3 < 1e-05:
                                all_combinations['Symmetric tilt'].append([list(val), miller2])
                                sym_tilt = True
                                break
                    if not sym_tilt:
                        all_combinations['Normal tilt'].append([list(val), miller2])
                else:
                    all_combinations['Mixed'].append([list(val), miller2])
    return all_combinations