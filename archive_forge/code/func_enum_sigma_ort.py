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
def enum_sigma_ort(cutoff, r_axis, c2_b2_a2_ratio):
    """
        Find all possible sigma values and corresponding rotation angles
        within a sigma value cutoff with known rotation axis in orthorhombic system.
        The algorithm for this code is from reference, Scipta Metallurgica 27, 291(1992).

        Args:
            cutoff (int): the cutoff of sigma values.
            r_axis (list of 3 integers, e.g. u, v, w):
                the rotation axis of the grain boundary, with the format of [u,v,w].
            c2_b2_a2_ratio (list of 3 integers, e.g. mu,lambda, mv):
                mu:lam:mv is the square of the orthorhombic axial ratio with rational
                numbers. If irrational for one axis, set it to None.
                e.g. mu:lam:mv = c2,None,a2, means b2 is irrational.

        Returns:
            dict: sigmas  dictionary with keys as the possible integer sigma values
                and values as list of the possible rotation angles to the
                corresponding sigma values. e.g. the format as
                {sigma1: [angle11,angle12,...], sigma2: [angle21, angle22,...],...}
                Note: the angles are the rotation angle of one grain respect to the
                other grain.
                When generate the microstructure of the grain boundary using these
                angles, you need to analyze the symmetry of the structure. Different
                angles may result in equivalent microstructures.
        """
    sigmas = {}
    if reduce(gcd, r_axis) != 1:
        r_axis = [int(round(x / reduce(gcd, r_axis))) for x in r_axis]
    u, v, w = r_axis
    if None in c2_b2_a2_ratio:
        mu, lam, mv = c2_b2_a2_ratio
        non_none = [i for i in c2_b2_a2_ratio if i is not None]
        if len(non_none) < 2:
            raise RuntimeError('No CSL exist for two irrational numbers')
        non1, non2 = non_none
        if reduce(gcd, non_none) != 1:
            temp = reduce(gcd, non_none)
            non1 = int(round(non1 / temp))
            non2 = int(round(non2 / temp))
        if mu is None:
            lam = non1
            mv = non2
            mu = 1
            if w != 0 and (u != 0 or v != 0):
                raise RuntimeError('For irrational c2, CSL only exist for [0,0,1] or [u,v,0] and m = 0')
        elif lam is None:
            mu = non1
            mv = non2
            lam = 1
            if v != 0 and (u != 0 or w != 0):
                raise RuntimeError('For irrational b2, CSL only exist for [0,1,0] or [u,0,w] and m = 0')
        elif mv is None:
            mu = non1
            lam = non2
            mv = 1
            if u != 0 and (w != 0 or v != 0):
                raise RuntimeError('For irrational a2, CSL only exist for [1,0,0] or [0,v,w] and m = 0')
    else:
        mu, lam, mv = c2_b2_a2_ratio
        if reduce(gcd, c2_b2_a2_ratio) != 1:
            temp = reduce(gcd, c2_b2_a2_ratio)
            mu = int(round(mu / temp))
            mv = int(round(mv / temp))
            lam = int(round(lam / temp))
        if u == 0 and v == 0:
            mu = 1
        if u == 0 and w == 0:
            lam = 1
        if v == 0 and w == 0:
            mv = 1
    d = (mv * u ** 2 + lam * v ** 2) * mv + w ** 2 * mu * mv
    n_max = int(np.sqrt(cutoff * 4 * mu * mv * mv * lam / d))
    for n in range(1, n_max + 1):
        mu_temp, lam_temp, mv_temp = c2_b2_a2_ratio
        if mu_temp is None and w == 0 or (lam_temp is None and v == 0) or (mv_temp is None and u == 0):
            m_max = 0
        else:
            m_max = int(np.sqrt((cutoff * 4 * mu * mv * lam * mv - n ** 2 * d) / mu / lam))
        for m in range(m_max + 1):
            if gcd(m, n) == 1 or m == 0:
                R_list = [(u ** 2 * mv * mv - lam * v ** 2 * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * lam * (v * u * mv * n ** 2 - w * mu * m * n), 2 * mu * (u * w * mv * n ** 2 + v * lam * m * n), 2 * mv * (u * v * mv * n ** 2 + w * mu * m * n), (v ** 2 * mv * lam - u ** 2 * mv * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * mv * mu * (v * w * n ** 2 - u * m * n), 2 * mv * (u * w * mv * n ** 2 - v * lam * m * n), 2 * lam * mv * (v * w * n ** 2 + u * m * n), (w ** 2 * mu * mv - u ** 2 * mv * mv - v ** 2 * mv * lam) * n ** 2 + lam * mu * m ** 2]
                m = -1 * m
                R_list_inv = [(u ** 2 * mv * mv - lam * v ** 2 * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * lam * (v * u * mv * n ** 2 - w * mu * m * n), 2 * mu * (u * w * mv * n ** 2 + v * lam * m * n), 2 * mv * (u * v * mv * n ** 2 + w * mu * m * n), (v ** 2 * mv * lam - u ** 2 * mv * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * mv * mu * (v * w * n ** 2 - u * m * n), 2 * mv * (u * w * mv * n ** 2 - v * lam * m * n), 2 * lam * mv * (v * w * n ** 2 + u * m * n), (w ** 2 * mu * mv - u ** 2 * mv * mv - v ** 2 * mv * lam) * n ** 2 + lam * mu * m ** 2]
                m = -1 * m
                F = mu * lam * m ** 2 + d * n ** 2
                all_list = R_list + R_list_inv + [F]
                com_fac = reduce(gcd, all_list)
                sigma = int(round((mu * lam * m ** 2 + d * n ** 2) / com_fac))
                if 1 < sigma <= cutoff:
                    if sigma not in list(sigmas):
                        angle = 180.0 if m == 0 else 2 * np.arctan(n / m * np.sqrt(d / mu / lam)) / np.pi * 180
                        sigmas[sigma] = [angle]
                    else:
                        angle = 180.0 if m == 0 else 2 * np.arctan(n / m * np.sqrt(d / mu / lam)) / np.pi * 180
                        if angle not in sigmas[sigma]:
                            sigmas[sigma].append(angle)
        if m_max == 0:
            break
    return sigmas