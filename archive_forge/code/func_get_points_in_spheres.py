from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
def get_points_in_spheres(all_coords: np.ndarray, center_coords: np.ndarray, r: float, pbc: bool | list[bool] | PbcLike=True, numerical_tol: float=1e-08, lattice: Lattice | None=None, return_fcoords: bool=False) -> list[list[tuple[np.ndarray, float, int, np.ndarray]]]:
    """For each point in `center_coords`, get all the neighboring points in `all_coords` that are within the
    cutoff radius `r`.

    Args:
        all_coords: (list of Cartesian coordinates) all available points
        center_coords: (list of Cartesian coordinates) all centering points
        r: (float) cutoff radius
        pbc: (bool or a list of bool) whether to set periodic boundaries
        numerical_tol: (float) numerical tolerance
        lattice: (Lattice) lattice to consider when PBC is enabled
        return_fcoords: (bool) whether to return fractional coords when pbc is set.

    Returns:
        List[List[Tuple[coords, distance, index, image]]]
    """
    if isinstance(pbc, bool):
        pbc = [pbc] * 3
    pbc = np.array(pbc, dtype=bool)
    if return_fcoords and lattice is None:
        raise ValueError('Lattice needs to be supplied to compute fractional coordinates')
    center_coords_min = np.min(center_coords, axis=0)
    center_coords_max = np.max(center_coords, axis=0)
    global_min = center_coords_min - r - numerical_tol
    global_max = center_coords_max + r + numerical_tol
    if np.any(pbc):
        if lattice is None:
            raise ValueError('Lattice needs to be supplied when considering periodic boundary')
        recp_len = np.array(lattice.reciprocal_lattice.abc)
        maxr = np.ceil((r + 0.15) * recp_len / (2 * math.pi))
        frac_coords = lattice.get_fractional_coords(center_coords)
        nmin_temp = np.floor(np.min(frac_coords, axis=0)) - maxr
        nmax_temp = np.ceil(np.max(frac_coords, axis=0)) + maxr
        nmin = np.zeros_like(nmin_temp)
        nmin[pbc] = nmin_temp[pbc]
        nmax = np.ones_like(nmax_temp)
        nmax[pbc] = nmax_temp[pbc]
        all_ranges = [np.arange(x, y, dtype='int64') for x, y in zip(nmin, nmax)]
        matrix = lattice.matrix
        image_offsets = lattice.get_fractional_coords(all_coords)
        all_fcoords = []
        for kk in range(3):
            if pbc[kk]:
                all_fcoords.append(np.mod(image_offsets[:, kk:kk + 1], 1))
            else:
                all_fcoords.append(image_offsets[:, kk:kk + 1])
        all_fcoords = np.concatenate(all_fcoords, axis=1)
        image_offsets = image_offsets - all_fcoords
        coords_in_cell = np.dot(all_fcoords, matrix)
        valid_coords = []
        valid_images = []
        valid_indices = []
        for image in itertools.product(*all_ranges):
            coords = np.dot(image, matrix) + coords_in_cell
            valid_index_bool = np.all(np.bitwise_and(coords > global_min[None, :], coords < global_max[None, :]), axis=1)
            ind = np.arange(len(all_coords))
            if np.any(valid_index_bool):
                valid_coords.append(coords[valid_index_bool])
                valid_images.append(np.tile(image, [np.sum(valid_index_bool), 1]) - image_offsets[valid_index_bool])
                valid_indices.extend([k for k in ind if valid_index_bool[k]])
        if len(valid_coords) < 1:
            return [[]] * len(center_coords)
        valid_coords = np.concatenate(valid_coords, axis=0)
        valid_images = np.concatenate(valid_images, axis=0)
    else:
        valid_coords = all_coords
        valid_images = [[0, 0, 0]] * len(valid_coords)
        valid_indices = np.arange(len(valid_coords))
    all_cube_index = _compute_cube_index(valid_coords, global_min, r)
    nx, ny, nz = _compute_cube_index(global_max, global_min, r) + 1
    all_cube_index = _three_to_one(all_cube_index, ny, nz)
    site_cube_index = _three_to_one(_compute_cube_index(center_coords, global_min, r), ny, nz)
    cube_to_coords: dict[int, list] = collections.defaultdict(list)
    cube_to_images: dict[int, list] = collections.defaultdict(list)
    cube_to_indices: dict[int, list] = collections.defaultdict(list)
    for ii, jj, kk, ll in zip(all_cube_index.ravel(), valid_coords, valid_images, valid_indices):
        cube_to_coords[ii].append(jj)
        cube_to_images[ii].append(kk)
        cube_to_indices[ii].append(ll)
    site_neighbors = find_neighbors(site_cube_index, nx, ny, nz)
    neighbors: list[list[tuple[np.ndarray, float, int, np.ndarray]]] = []
    for ii, jj in zip(center_coords, site_neighbors):
        l1 = np.array(_three_to_one(jj, ny, nz), dtype=int).ravel()
        ks = [k for k in l1 if k in cube_to_coords]
        if not ks:
            neighbors.append([])
            continue
        nn_coords = np.concatenate([cube_to_coords[k] for k in ks], axis=0)
        nn_images = itertools.chain(*(cube_to_images[k] for k in ks))
        nn_indices = itertools.chain(*(cube_to_indices[k] for k in ks))
        distances = np.linalg.norm(nn_coords - ii[None, :], axis=1)
        nns: list[tuple[np.ndarray, float, int, np.ndarray]] = []
        for coord, index, image, dist in zip(nn_coords, nn_indices, nn_images, distances):
            if dist < r + numerical_tol:
                if return_fcoords and lattice is not None:
                    coord = np.round(lattice.get_fractional_coords(coord), 10)
                nn = (coord, float(dist), int(index), image)
                nns.append(nn)
        neighbors.append(nns)
    return neighbors