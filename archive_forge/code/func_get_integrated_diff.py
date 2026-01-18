from __future__ import annotations
import itertools
import json
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from scipy.interpolate import RegularGridInterpolator
from pymatgen.core import Element, Site, Structure
from pymatgen.core.units import ang_to_bohr, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
def get_integrated_diff(self, ind, radius, nbins=1):
    """
        Get integrated difference of atom index ind up to radius. This can be
        an extremely computationally intensive process, depending on how many
        grid points are in the VolumetricData.

        Args:
            ind (int): Index of atom.
            radius (float): Radius of integration.
            nbins (int): Number of bins. Defaults to 1. This allows one to
                obtain the charge integration up to a list of the cumulative
                charge integration values for radii for [radius/nbins,
                2 * radius/nbins, ....].

        Returns:
            Differential integrated charge as a np array of [[radius, value],
            ...]. Format is for ease of plotting. E.g., plt.plot(data[:,0],
            data[:,1])
        """
    if not self.is_spin_polarized:
        radii = [radius / nbins * (i + 1) for i in range(nbins)]
        data = np.zeros((nbins, 2))
        data[:, 0] = radii
        return data
    struct = self.structure
    a = self.dim
    if ind not in self._distance_matrix or self._distance_matrix[ind]['max_radius'] < radius:
        coords = []
        for x, y, z in itertools.product(*(list(range(i)) for i in a)):
            coords.append([x / a[0], y / a[1], z / a[2]])
        sites_dist = struct.lattice.get_points_in_sphere(coords, struct[ind].coords, radius)
        self._distance_matrix[ind] = {'max_radius': radius, 'data': np.array(sites_dist, dtype=object)}
    data = self._distance_matrix[ind]['data']
    inds = data[:, 1] <= radius
    dists = data[inds, 1]
    data_inds = np.rint(np.mod(list(data[inds, 0]), 1) * np.tile(a, (len(dists), 1))).astype(int)
    vals = [self.data['diff'][x, y, z] for x, y, z in data_inds]
    hist, edges = np.histogram(dists, bins=nbins, range=[0, radius], weights=vals)
    data = np.zeros((nbins, 2))
    data[:, 0] = edges[1:]
    data[:, 1] = [sum(hist[0:i + 1]) / self.ngridpts for i in range(nbins)]
    return data