from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
@staticmethod
def get_pourbaix_domains(pourbaix_entries, limits=None):
    """
        Returns a set of Pourbaix stable domains (i. e. polygons) in
        pH-V space from a list of pourbaix_entries.

        This function works by using scipy's HalfspaceIntersection
        function to construct all of the 2-D polygons that form the
        boundaries of the planes corresponding to individual entry
        gibbs free energies as a function of pH and V. Hyperplanes
        of the form a*pH + b*V + 1 - g(0, 0) are constructed and
        supplied to HalfspaceIntersection, which then finds the
        boundaries of each Pourbaix region using the intersection
        points.

        Args:
            pourbaix_entries ([PourbaixEntry]): Pourbaix entries
                with which to construct stable Pourbaix domains
            limits ([[float]]): limits in which to do the pourbaix
                analysis

        Returns:
            Returns a dict of the form {entry: [boundary_points]}.
            The list of boundary points are the sides of the N-1
            dim polytope bounding the allowable ph-V range of each entry.
        """
    if limits is None:
        limits = [[-2, 16], [-4, 4]]
    hyperplanes = [np.array([-PREFAC * entry.npH, -entry.nPhi, 0, -entry.energy]) * entry.normalization_factor for entry in pourbaix_entries]
    hyperplanes = np.array(hyperplanes)
    hyperplanes[:, 2] = 1
    max_contribs = np.max(np.abs(hyperplanes), axis=0)
    g_max = np.dot(-max_contribs, [limits[0][1], limits[1][1], 0, 1])
    border_hyperplanes = [[-1, 0, 0, limits[0][0]], [1, 0, 0, -limits[0][1]], [0, -1, 0, limits[1][0]], [0, 1, 0, -limits[1][1]], [0, 0, -1, 2 * g_max]]
    hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
    interior_point = [*np.average(limits, axis=1).tolist(), g_max]
    hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))
    pourbaix_domains = {entry: [] for entry in pourbaix_entries}
    for intersection, facet in zip(hs_int.intersections, hs_int.dual_facets):
        for v in facet:
            if v < len(pourbaix_entries):
                this_entry = pourbaix_entries[v]
                pourbaix_domains[this_entry].append(intersection)
    pourbaix_domains = {k: v for k, v in pourbaix_domains.items() if v}
    pourbaix_domain_vertices = {}
    for entry, points in pourbaix_domains.items():
        points = np.array(points)[:, :2]
        points = points[np.lexsort(np.transpose(points))]
        center = np.average(points, axis=0)
        points_centered = points - center
        points_centered = sorted(points_centered, key=cmp_to_key(lambda x, y: x[0] * y[1] - x[1] * y[0]))
        points = points_centered + center
        simplices = [Simplex(points[indices]) for indices in ConvexHull(points).simplices]
        pourbaix_domains[entry] = simplices
        pourbaix_domain_vertices[entry] = points
    return (pourbaix_domains, pourbaix_domain_vertices)