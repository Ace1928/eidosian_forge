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
def enum_sigma_rho(cutoff, r_axis, ratio_alpha):
    """
        Find all possible sigma values and corresponding rotation angles
        within a sigma value cutoff with known rotation axis in rhombohedral system.
        The algorithm for this code is from reference, Acta Cryst, A45,505(1989).

        Args:
            cutoff (int): the cutoff of sigma values.
            r_axis (list[int]): of 3 integers, e.g. u, v, w
                    or 4 integers, e.g. u, v, t, w):
                    the rotation axis of the grain boundary, with the format of [u,v,w]
                    or Weber indices [u, v, t, w].
            ratio_alpha (list of two integers, e.g. mu, mv):
                    mu/mv is the ratio of (1+2*cos(alpha))/cos(alpha) with rational number.
                    If irrational, set ratio_alpha = None.

        Returns:
            sigmas (dict):
                    dictionary with keys as the possible integer sigma values
                    and values as list of the possible rotation angles to the
                    corresponding sigma values.
                    e.g. the format as
                    {sigma1: [angle11,angle12,...], sigma2: [angle21, angle22,...],...}
                    Note: the angles are the rotation angle of one grain respect to the
                    other grain.
                    When generate the microstructure of the grain boundary using these
                    angles, you need to analyze the symmetry of the structure. Different
                    angles may result in equivalent microstructures.
        """
    sigmas = {}
    if len(r_axis) == 4:
        u1 = r_axis[0]
        v1 = r_axis[1]
        w1 = r_axis[3]
        u = 2 * u1 + v1 + w1
        v = v1 + w1 - u1
        w = w1 - 2 * v1 - u1
        r_axis = [u, v, w]
    if reduce(gcd, r_axis) != 1:
        r_axis = [int(round(x / reduce(gcd, r_axis))) for x in r_axis]
    u, v, w = r_axis
    if ratio_alpha is None:
        mu, mv = [1, 1]
        if u + v + w != 0 and (u != v or u != w):
            raise RuntimeError('For irrational ratio_alpha, CSL only exist for [1,1,1] or [u, v, -(u+v)] and m =0')
    else:
        mu, mv = ratio_alpha
        if gcd(mu, mv) != 1:
            temp = gcd(mu, mv)
            mu = int(round(mu / temp))
            mv = int(round(mv / temp))
    d = (u ** 2 + v ** 2 + w ** 2) * (mu - 2 * mv) + 2 * mv * (v * w + w * u + u * v)
    n_max = int(np.sqrt(cutoff * abs(4 * mu * (mu - 3 * mv)) / abs(d)))
    for n in range(1, n_max + 1):
        if ratio_alpha is None and u + v + w == 0:
            m_max = 0
        else:
            m_max = int(np.sqrt((cutoff * abs(4 * mu * (mu - 3 * mv)) - n ** 2 * d) / mu))
        for m in range(m_max + 1):
            if gcd(m, n) == 1 or m == 0:
                R_list = [(mu - 2 * mv) * (u ** 2 - v ** 2 - w ** 2) * n ** 2 + 2 * mv * (v - w) * m * n - 2 * mv * v * w * n ** 2 + mu * m ** 2, 2 * (mv * u * n * (w * n + u * n - m) - (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), 2 * (mv * u * n * (v * n + u * n + m) + (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * v * n * (w * n + v * n + m) + (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), (mu - 2 * mv) * (v ** 2 - w ** 2 - u ** 2) * n ** 2 + 2 * mv * (w - u) * m * n - 2 * mv * u * w * n ** 2 + mu * m ** 2, 2 * (mv * v * n * (v * n + u * n - m) - (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), 2 * (mv * w * n * (w * n + v * n - m) - (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * w * n * (w * n + u * n + m) + (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), (mu - 2 * mv) * (w ** 2 - u ** 2 - v ** 2) * n ** 2 + 2 * mv * (u - v) * m * n - 2 * mv * u * v * n ** 2 + mu * m ** 2]
                m = -1 * m
                R_list_inv = [(mu - 2 * mv) * (u ** 2 - v ** 2 - w ** 2) * n ** 2 + 2 * mv * (v - w) * m * n - 2 * mv * v * w * n ** 2 + mu * m ** 2, 2 * (mv * u * n * (w * n + u * n - m) - (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), 2 * (mv * u * n * (v * n + u * n + m) + (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * v * n * (w * n + v * n + m) + (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), (mu - 2 * mv) * (v ** 2 - w ** 2 - u ** 2) * n ** 2 + 2 * mv * (w - u) * m * n - 2 * mv * u * w * n ** 2 + mu * m ** 2, 2 * (mv * v * n * (v * n + u * n - m) - (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), 2 * (mv * w * n * (w * n + v * n - m) - (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * w * n * (w * n + u * n + m) + (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), (mu - 2 * mv) * (w ** 2 - u ** 2 - v ** 2) * n ** 2 + 2 * mv * (u - v) * m * n - 2 * mv * u * v * n ** 2 + mu * m ** 2]
                m = -1 * m
                F = mu * m ** 2 + d * n ** 2
                all_list = R_list_inv + R_list + [F]
                com_fac = reduce(gcd, all_list)
                sigma = int(round(abs(F / com_fac)))
                if 1 < sigma <= cutoff:
                    if sigma not in list(sigmas):
                        angle = 180.0 if m == 0 else 2 * np.arctan(n / m * np.sqrt(d / mu)) / np.pi * 180
                        sigmas[sigma] = [angle]
                    else:
                        angle = 180 if m == 0 else 2 * np.arctan(n / m * np.sqrt(d / mu)) / np.pi * 180.0
                        if angle not in sigmas[sigma]:
                            sigmas[sigma].append(angle)
        if m_max == 0:
            break
    return sigmas