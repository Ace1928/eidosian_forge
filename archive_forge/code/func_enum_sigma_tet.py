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
def enum_sigma_tet(cutoff, r_axis, c2_a2_ratio):
    """
        Find all possible sigma values and corresponding rotation angles
        within a sigma value cutoff with known rotation axis in tetragonal system.
        The algorithm for this code is from reference, Acta Cryst, B46,117(1990).

        Args:
            cutoff (int): the cutoff of sigma values.
            r_axis (list of 3 integers, e.g. u, v, w):
                the rotation axis of the grain boundary, with the format of [u,v,w].
            c2_a2_ratio (list of two integers, e.g. mu, mv):
                mu/mv is the square of the tetragonal axial ratio with rational number.
                if irrational, set c2_a2_ratio = None

        Returns:
            dict: sigmas dictionary with keys as the possible integer sigma values
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
    if c2_a2_ratio is None:
        mu, mv = [1, 1]
        if w != 0 and (u != 0 or v != 0):
            raise RuntimeError('For irrational c2/a2, CSL only exist for [0,0,1] or [u,v,0] and m = 0')
    else:
        mu, mv = c2_a2_ratio
        if gcd(mu, mv) != 1:
            temp = gcd(mu, mv)
            mu = int(round(mu / temp))
            mv = int(round(mv / temp))
    d = (u ** 2 + v ** 2) * mv + w ** 2 * mu
    n_max = int(np.sqrt(cutoff * 4 * mu * mv / d))
    for n in range(1, n_max + 1):
        m_max = 0 if c2_a2_ratio is None and w == 0 else int(np.sqrt((cutoff * 4 * mu * mv - n ** 2 * d) / mu))
        for m in range(m_max + 1):
            if gcd(m, n) == 1 or m == 0:
                R_list = [(u ** 2 * mv - v ** 2 * mv - w ** 2 * mu) * n ** 2 + mu * m ** 2, 2 * v * u * mv * n ** 2 - 2 * w * mu * m * n, 2 * u * w * mu * n ** 2 + 2 * v * mu * m * n, 2 * u * v * mv * n ** 2 + 2 * w * mu * m * n, (v ** 2 * mv - u ** 2 * mv - w ** 2 * mu) * n ** 2 + mu * m ** 2, 2 * v * w * mu * n ** 2 - 2 * u * mu * m * n, 2 * u * w * mv * n ** 2 - 2 * v * mv * m * n, 2 * v * w * mv * n ** 2 + 2 * u * mv * m * n, (w ** 2 * mu - u ** 2 * mv - v ** 2 * mv) * n ** 2 + mu * m ** 2]
                m = -1 * m
                R_list_inv = [(u ** 2 * mv - v ** 2 * mv - w ** 2 * mu) * n ** 2 + mu * m ** 2, 2 * v * u * mv * n ** 2 - 2 * w * mu * m * n, 2 * u * w * mu * n ** 2 + 2 * v * mu * m * n, 2 * u * v * mv * n ** 2 + 2 * w * mu * m * n, (v ** 2 * mv - u ** 2 * mv - w ** 2 * mu) * n ** 2 + mu * m ** 2, 2 * v * w * mu * n ** 2 - 2 * u * mu * m * n, 2 * u * w * mv * n ** 2 - 2 * v * mv * m * n, 2 * v * w * mv * n ** 2 + 2 * u * mv * m * n, (w ** 2 * mu - u ** 2 * mv - v ** 2 * mv) * n ** 2 + mu * m ** 2]
                m = -1 * m
                F = mu * m ** 2 + d * n ** 2
                all_list = R_list + R_list_inv + [F]
                com_fac = reduce(gcd, all_list)
                sigma = int(round((mu * m ** 2 + d * n ** 2) / com_fac))
                if 1 < sigma <= cutoff:
                    if sigma not in list(sigmas):
                        angle = 180.0 if m == 0 else 2 * np.arctan(n / m * np.sqrt(d / mu)) / np.pi * 180
                        sigmas[sigma] = [angle]
                    else:
                        angle = 180.0 if m == 0 else 2 * np.arctan(n / m * np.sqrt(d / mu)) / np.pi * 180
                        if angle not in sigmas[sigma]:
                            sigmas[sigma].append(angle)
        if m_max == 0:
            break
    return sigmas