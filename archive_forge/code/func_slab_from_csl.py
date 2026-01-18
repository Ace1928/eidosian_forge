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
def slab_from_csl(csl, surface, normal, trans_cry, max_search=20, quick_gen=False):
    """
        By linear operation of csl lattice vectors to get the best corresponding
        slab lattice. That is the area of a,b vectors (within the surface plane)
        is the smallest, the c vector first, has shortest length perpendicular
        to surface [h,k,l], second, has shortest length itself.

        Args:
            csl (3 by 3 integer array):
                    input csl lattice.
            surface (list of 3 integers, e.g. h, k, l):
                    the miller index of the surface, with the format of [h,k,l]
            normal (logic):
                    determine if the c vector needs to perpendicular to surface
            trans_cry (3 by 3 array):
                    transform matrix from crystal system to orthogonal system
            max_search (int): max search for the GB lattice vectors that give the smallest GB
                lattice. If normal is true, also max search the GB c vector that perpendicular
                to the plane.
            quick_gen (bool): whether to quickly generate a supercell, no need to find the smallest
                cell if set to true.

        Returns:
            t_matrix: a slab lattice ( 3 by 3 integer array):
        """
    trans = trans_cry
    ctrans = np.linalg.inv(trans.T)
    t_matrix = csl.copy()
    ab_vector = []
    miller = np.matmul(surface, csl.T)
    if reduce(gcd, miller) != 1:
        miller = [int(round(x / reduce(gcd, miller))) for x in miller]
    miller_nonzero = []
    if quick_gen:
        scale_factor = []
        eye = np.eye(3, dtype=int)
        for ii, jj in enumerate(miller):
            if jj == 0:
                scale_factor.append(eye[ii])
            else:
                miller_nonzero.append(ii)
        if len(scale_factor) < 2:
            index_len = len(miller_nonzero)
            for ii in range(index_len):
                for jj in range(ii + 1, index_len):
                    lcm_miller = lcm(miller[miller_nonzero[ii]], miller[miller_nonzero[jj]])
                    scl_factor = [0, 0, 0]
                    scl_factor[miller_nonzero[ii]] = -int(round(lcm_miller / miller[miller_nonzero[ii]]))
                    scl_factor[miller_nonzero[jj]] = int(round(lcm_miller / miller[miller_nonzero[jj]]))
                    scale_factor.append(scl_factor)
                    if len(scale_factor) == 2:
                        break
        t_matrix[0] = np.array(np.dot(scale_factor[0], csl))
        t_matrix[1] = np.array(np.dot(scale_factor[1], csl))
        t_matrix[2] = csl[miller_nonzero[0]]
        if abs(np.linalg.det(t_matrix)) > 1000:
            warnings.warn('Too large matrix. Suggest to use quick_gen=False')
        return t_matrix
    for ii, jj in enumerate(miller):
        if jj == 0:
            ab_vector.append(csl[ii])
        else:
            c_index = ii
            miller_nonzero.append(jj)
    if len(miller_nonzero) > 1:
        t_matrix[2] = csl[c_index]
        index_len = len(miller_nonzero)
        lcm_miller = []
        for ii in range(index_len):
            for jj in range(ii + 1, index_len):
                com_gcd = gcd(miller_nonzero[ii], miller_nonzero[jj])
                mil1 = int(round(miller_nonzero[ii] / com_gcd))
                mil2 = int(round(miller_nonzero[jj] / com_gcd))
                lcm_miller.append(max(abs(mil1), abs(mil2)))
        lcm_sorted = sorted(lcm_miller)
        max_j = lcm_sorted[0] if index_len == 2 else lcm_sorted[1]
    else:
        if not normal:
            t_matrix[0] = ab_vector[0]
            t_matrix[1] = ab_vector[1]
            t_matrix[2] = csl[c_index]
            return t_matrix
        max_j = abs(miller_nonzero[0])
    max_j = min(max_j, max_search)
    area = None
    c_norm = np.linalg.norm(np.matmul(t_matrix[2], trans))
    c_length = np.abs(np.dot(t_matrix[2], surface))
    if normal:
        c_cross = np.cross(np.matmul(t_matrix[2], trans), np.matmul(surface, ctrans))
        normal_init = np.linalg.norm(c_cross) < 1e-08
    jj = np.arange(0, max_j + 1)
    combination = []
    for ii in product(jj, repeat=3):
        if sum(abs(np.array(ii))) != 0:
            combination.append(list(ii))
        if len(np.nonzero(ii)[0]) == 3:
            for i1 in range(3):
                new_i = list(ii).copy()
                new_i[i1] = -1 * new_i[i1]
                combination.append(new_i)
        elif len(np.nonzero(ii)[0]) == 2:
            new_i = list(ii).copy()
            new_i[np.nonzero(ii)[0][0]] = -1 * new_i[np.nonzero(ii)[0][0]]
            combination.append(new_i)
    for ii in combination:
        if reduce(gcd, ii) == 1:
            temp = np.dot(np.array(ii), csl)
            if abs(np.dot(temp, surface) - 0) < 1e-08:
                ab_vector.append(temp)
            else:
                c_len_temp = np.abs(np.dot(temp, surface))
                c_norm_temp = np.linalg.norm(np.matmul(temp, trans))
                if normal:
                    c_cross = np.cross(np.matmul(temp, trans), np.matmul(surface, ctrans))
                    if np.linalg.norm(c_cross) < 1e-08:
                        if normal_init:
                            if c_norm_temp < c_norm:
                                t_matrix[2] = temp
                                c_norm = c_norm_temp
                        else:
                            c_norm = c_norm_temp
                            normal_init = True
                            t_matrix[2] = temp
                elif c_len_temp < c_length or (abs(c_len_temp - c_length) < 1e-08 and c_norm_temp < c_norm):
                    t_matrix[2] = temp
                    c_norm = c_norm_temp
                    c_length = c_len_temp
    if normal and (not normal_init):
        logger.info('Did not find the perpendicular c vector, increase max_j')
        while not normal_init:
            if max_j == max_search:
                warnings.warn('Cannot find the perpendicular c vector, please increase max_search')
                break
            max_j = 3 * max_j
            max_j = min(max_j, max_search)
            jj = np.arange(0, max_j + 1)
            combination = []
            for ii in product(jj, repeat=3):
                if sum(abs(np.array(ii))) != 0:
                    combination.append(list(ii))
                if len(np.nonzero(ii)[0]) == 3:
                    for i1 in range(3):
                        new_i = list(ii).copy()
                        new_i[i1] = -1 * new_i[i1]
                        combination.append(new_i)
                elif len(np.nonzero(ii)[0]) == 2:
                    new_i = list(ii).copy()
                    new_i[np.nonzero(ii)[0][0]] = -1 * new_i[np.nonzero(ii)[0][0]]
                    combination.append(new_i)
            for ii in combination:
                if reduce(gcd, ii) == 1:
                    temp = np.dot(np.array(ii), csl)
                    if abs(np.dot(temp, surface) - 0) > 1e-08:
                        c_cross = np.cross(np.matmul(temp, trans), np.matmul(surface, ctrans))
                        if np.linalg.norm(c_cross) < 1e-08:
                            c_norm_temp = np.linalg.norm(np.matmul(temp, trans))
                            if normal_init:
                                if c_norm_temp < c_norm:
                                    t_matrix[2] = temp
                                    c_norm = c_norm_temp
                            else:
                                c_norm = c_norm_temp
                                normal_init = True
                                t_matrix[2] = temp
            if normal_init:
                logger.info('Found perpendicular c vector')
    for ii in combinations(ab_vector, 2):
        area_temp = np.linalg.norm(np.cross(np.matmul(ii[0], trans), np.matmul(ii[1], trans)))
        if abs(area_temp - 0) > 1e-08:
            ab_norm_temp = np.linalg.norm(np.matmul(ii[0], trans)) + np.linalg.norm(np.matmul(ii[1], trans))
            if area is None:
                area = area_temp
                ab_norm = ab_norm_temp
                t_matrix[0] = ii[0]
                t_matrix[1] = ii[1]
            elif area_temp < area or (abs(area - area_temp) < 1e-08 and ab_norm_temp < ab_norm):
                t_matrix[0] = ii[0]
                t_matrix[1] = ii[1]
                area = area_temp
                ab_norm = ab_norm_temp
    if np.linalg.det(np.matmul(t_matrix, trans)) < 0:
        t_matrix *= -1
    if normal and abs(np.linalg.det(t_matrix)) > 1000:
        warnings.warn('Too large matrix. Suggest to use Normal=False')
    return t_matrix