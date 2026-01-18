from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.math_utils import normal_cdf_step
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
def setup_voronoi_list(self, indices, voronoi_cutoff):
    """
        Set up of the voronoi list of neighbors by calling qhull.

        Args:
            indices: indices of the sites for which the Voronoi is needed.
            voronoi_cutoff: Voronoi cutoff for the search of neighbors.

        Raises:
            RuntimeError: If an infinite vertex is found in the voronoi construction.
        """
    self.voronoi_list2 = [None] * len(self.structure)
    self.voronoi_list_coords = [None] * len(self.structure)
    logging.debug('Getting all neighbors in structure')
    struct_neighbors = self.structure.get_all_neighbors(voronoi_cutoff, include_index=True)
    size_neighbors = [not len(neigh) > 3 for neigh in struct_neighbors]
    if np.any(size_neighbors):
        logging.debug('Please consider increasing voronoi_distance_cutoff')
    t1 = time.process_time()
    logging.debug('Setting up Voronoi list :')
    for jj, isite in enumerate(indices, start=1):
        logging.debug(f'  - Voronoi analysis for site #{isite} ({jj}/{len(indices)})')
        site = self.structure[isite]
        neighbors1 = [(site, 0.0, isite)]
        neighbors1.extend(struct_neighbors[isite])
        distances = [i[1] for i in sorted(neighbors1, key=lambda s: s[1])]
        neighbors = [i[0] for i in sorted(neighbors1, key=lambda s: s[1])]
        qvoronoi_input = [s.coords for s in neighbors]
        voro = Voronoi(points=qvoronoi_input, qhull_options='o Fv')
        all_vertices = voro.vertices
        results2 = []
        max_angle = 0.0
        min_dist = 10000.0
        for idx, ridge_points in enumerate(voro.ridge_points):
            if 0 in ridge_points:
                ridge_vertices_indices = voro.ridge_vertices[idx]
                if -1 in ridge_vertices_indices:
                    raise RuntimeError('This structure is pathological, infinite vertex in the voronoi construction')
                ridge_point2 = max(ridge_points)
                facets = [all_vertices[i] for i in ridge_vertices_indices]
                sa = solid_angle(site.coords, facets)
                max_angle = max([sa, max_angle])
                min_dist = min([min_dist, distances[ridge_point2]])
                for iii, sss in enumerate(self.structure):
                    if neighbors[ridge_point2].is_periodic_image(sss, tolerance=1e-06):
                        idx = iii
                        break
                results2.append({'site': neighbors[ridge_point2], 'angle': sa, 'distance': distances[ridge_point2], 'index': idx})
        for dd in results2:
            dd['normalized_angle'] = dd['angle'] / max_angle
            dd['normalized_distance'] = dd['distance'] / min_dist
        self.voronoi_list2[isite] = results2
        self.voronoi_list_coords[isite] = np.array([dd['site'].coords for dd in results2])
    t2 = time.process_time()
    logging.debug(f'Voronoi list set up in {t2 - t1:.2f} seconds')