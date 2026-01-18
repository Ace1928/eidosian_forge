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
def enum_sigma_cubic(cutoff, r_axis):
    """
        Find all possible sigma values and corresponding rotation angles
        within a sigma value cutoff with known rotation axis in cubic system.
        The algorithm for this code is from reference, Acta Cryst, A40,108(1984).

        Args:
            cutoff (int): the cutoff of sigma values.
            r_axis (list of 3 integers, e.g. u, v, w):
                the rotation axis of the grain boundary, with the format of [u,v,w].

        Returns:
            dict: sigmas dictionary with keys as the possible integer sigma values
                and values as list of the possible rotation angles to the
                corresponding sigma values. e.g. the format as
                {sigma1: [angle11,angle12,...], sigma2: [angle21, angle22,...],...}
                Note: the angles are the rotation angles of one grain respect to
                the other grain.
                When generate the microstructures of the grain boundary using these angles,
                you need to analyze the symmetry of the structure. Different angles may
                result in equivalent microstructures.
        """
    sigmas = {}
    if reduce(gcd, r_axis) != 1:
        r_axis = [int(round(x / reduce(gcd, r_axis))) for x in r_axis]
    odd_r = len(list(filter(lambda x: x % 2 == 1, r_axis)))
    if odd_r == 3:
        a_max = 4
    elif odd_r == 0:
        a_max = 1
    else:
        a_max = 2
    n_max = int(np.sqrt(cutoff * a_max / sum(np.array(r_axis) ** 2)))
    for n_loop in range(1, n_max + 1):
        n = n_loop
        m_max = int(np.sqrt(cutoff * a_max - n ** 2 * sum(np.array(r_axis) ** 2)))
        for m in range(m_max + 1):
            if gcd(m, n) == 1 or m == 0:
                n = 1 if m == 0 else n_loop
                quadruple = [m] + [x * n for x in r_axis]
                odd_qua = len(list(filter(lambda x: x % 2 == 1, quadruple)))
                if odd_qua == 4:
                    a = 4
                elif odd_qua == 2:
                    a = 2
                else:
                    a = 1
                sigma = int(round((m ** 2 + n ** 2 * sum(np.array(r_axis) ** 2)) / a))
                if 1 < sigma <= cutoff:
                    if sigma not in list(sigmas):
                        if m == 0:
                            angle = 180.0
                        else:
                            angle = 2 * np.arctan(n * np.sqrt(sum(np.array(r_axis) ** 2)) / m) / np.pi * 180
                        sigmas[sigma] = [angle]
                    else:
                        if m == 0:
                            angle = 180.0
                        else:
                            angle = 2 * np.arctan(n * np.sqrt(sum(np.array(r_axis) ** 2)) / m) / np.pi * 180
                        if angle not in sigmas[sigma]:
                            sigmas[sigma].append(angle)
    return sigmas