from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def get_primitive_structure(self, tolerance: float=0.25, use_site_props: bool=False, constrain_latt: list | dict | None=None):
    """This finds a smaller unit cell than the input. Sometimes it doesn"t
        find the smallest possible one, so this method is recursively called
        until it is unable to find a smaller cell.

        NOTE: if the tolerance is greater than 1/2 the minimum inter-site
        distance in the primitive cell, the algorithm will reject this lattice.

        Args:
            tolerance (float): Tolerance for each coordinate of a
                particular site in Angstroms. For example, [0.1, 0, 0.1] in cartesian
                coordinates will be considered to be on the same coordinates
                as [0, 0, 0] for a tolerance of 0.25. Defaults to 0.25.
            use_site_props (bool): Whether to account for site properties in
                differentiating sites.
            constrain_latt (list/dict): List of lattice parameters we want to
                preserve, e.g. ["alpha", "c"] or dict with the lattice
                parameter names as keys and values we want the parameters to
                be e.g. {"alpha": 90, "c": 2.5}.

        Returns:
            The most primitive structure found.
        """
    if constrain_latt is None:
        constrain_latt = []

    def site_label(site):
        if not use_site_props:
            return site.species_string
        parts = [site.species_string]
        for key in sorted(site.properties):
            parts.append(f'{key}={site.properties[key]}')
        return ', '.join(parts)
    sites = sorted(self._sites, key=site_label)
    grouped_sites = [list(a[1]) for a in itertools.groupby(sites, key=site_label)]
    grouped_fcoords = [np.array([s.frac_coords for s in g]) for g in grouped_sites]
    min_fcoords = min(grouped_fcoords, key=len)
    min_vecs = min_fcoords - min_fcoords[0]
    super_ftol = np.divide(tolerance, self.lattice.abc)
    super_ftol_2 = super_ftol * 2

    def pbc_coord_intersection(fc1, fc2, tol):
        """Returns the fractional coords in fc1 that have coordinates
            within tolerance to some coordinate in fc2.
            """
        dist = fc1[:, None, :] - fc2[None, :, :]
        dist -= np.round(dist)
        return fc1[np.any(np.all(dist < tol, axis=-1), axis=-1)]
    for group in sorted(grouped_fcoords, key=len):
        for frac_coords in group:
            min_vecs = pbc_coord_intersection(min_vecs, group - frac_coords, super_ftol_2)

    def get_hnf(fu):
        """Returns all possible distinct supercell matrices given a
            number of formula units in the supercell. Batches the matrices
            by the values in the diagonal (for less numpy overhead).
            Computational complexity is O(n^3), and difficult to improve.
            Might be able to do something smart with checking combinations of a
            and b first, though unlikely to reduce to O(n^2).
            """

        def factors(n: int):
            for idx in range(1, n + 1):
                if n % idx == 0:
                    yield idx
        for det in factors(fu):
            if det == 1:
                continue
            for a in factors(det):
                for e in factors(det // a):
                    g = det // a // e
                    yield (det, np.array([[[a, b, c], [0, e, f], [0, 0, g]] for b, c, f in itertools.product(range(a), range(a), range(e))]))
    grouped_non_nbrs = []
    for gfcoords in grouped_fcoords:
        fdist = gfcoords[None, :, :] - gfcoords[:, None, :]
        fdist -= np.round(fdist)
        np.abs(fdist, fdist)
        non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
        np.fill_diagonal(non_nbrs, val=True)
        grouped_non_nbrs.append(non_nbrs)
    num_fu = functools.reduce(math.gcd, map(len, grouped_sites))
    for size, ms in get_hnf(num_fu):
        inv_ms = np.linalg.inv(ms)
        dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
        dist -= np.round(dist)
        np.abs(dist, dist)
        is_close = np.all(dist < super_ftol, axis=-1)
        any_close = np.any(is_close, axis=-1)
        inds = np.all(any_close, axis=-1)
        for inv_m, m in zip(inv_ms[inds], ms[inds]):
            new_m = np.dot(inv_m, self.lattice.matrix)
            ftol = np.divide(tolerance, np.sqrt(np.sum(new_m ** 2, axis=1)))
            valid = True
            new_coords = []
            new_sp = []
            new_props = collections.defaultdict(list)
            new_labels = []
            for gsites, gfcoords, non_nbrs in zip(grouped_sites, grouped_fcoords, grouped_non_nbrs):
                all_frac = np.dot(gfcoords, m)
                fdist = all_frac[None, :, :] - all_frac[:, None, :]
                fdist = np.abs(fdist - np.round(fdist))
                close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
                groups = np.logical_and(close_in_prim, non_nbrs)
                if not np.all(np.sum(groups, axis=0) == size):
                    valid = False
                    break
                for group in groups:
                    if not np.all(groups[group][:, group]):
                        valid = False
                        break
                if not valid:
                    break
                added = np.zeros(len(gsites))
                new_fcoords = all_frac % 1
                for i, group in enumerate(groups):
                    if not added[i]:
                        added[group] = True
                        inds = np.where(group)[0]
                        coords = new_fcoords[inds[0]]
                        for n, j in enumerate(inds[1:]):
                            offset = new_fcoords[j] - coords
                            coords += (offset - np.round(offset)) / (n + 2)
                        new_sp.append(gsites[inds[0]].species)
                        for k in gsites[inds[0]].properties:
                            new_props[k].append(gsites[inds[0]].properties[k])
                        new_labels.append(gsites[inds[0]].label)
                        new_coords.append(coords)
            if valid:
                inv_m = np.linalg.inv(m)
                new_latt = Lattice(np.dot(inv_m, self.lattice.matrix))
                struct = Structure(new_latt, new_sp, new_coords, site_properties=new_props, labels=new_labels, coords_are_cartesian=False)
                p = struct.get_primitive_structure(tolerance=tolerance, use_site_props=use_site_props, constrain_latt=constrain_latt).get_reduced_structure()
                if not constrain_latt:
                    return p
                prim_latt, self_latt = (p.lattice, self.lattice)
                keys = tuple(constrain_latt)
                is_dict = isinstance(constrain_latt, dict)
                if np.allclose([getattr(prim_latt, key) for key in keys], [constrain_latt[key] if is_dict else getattr(self_latt, key) for key in keys]):
                    return p
    return self.copy()