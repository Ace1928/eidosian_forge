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
@classmethod
def from_slabs(cls, substrate_slab: Slab, film_slab: Slab, in_plane_offset: tuple[float, float]=(0, 0), gap: float=1.6, vacuum_over_film: float=0, interface_properties: dict | None=None, center_slab: bool=True) -> Self:
    """Makes an interface structure by merging a substrate and film slabs
        The film a- and b-vectors will be forced to be the substrate slab's
        a- and b-vectors.

        For now, it's suggested to use a factory method that will ensure the
        appropriate interface structure is already met.

        Args:
            substrate_slab (Slab): slab for the substrate
            film_slab (Slab): slab for the film
            in_plane_offset (tuple): fractional shift in plane for the film with respect to the substrate.
                For example, (0.5, 0.5) will shift the film by half the substrate's a- and b-vectors.
                Defaults to (0, 0).
            gap (float): gap between substrate and film in Angstroms. Defaults to 1.6.
            vacuum_over_film (float): vacuum space above the film in Angstroms. Defaults to 0.
            interface_properties (dict): misc properties to assign to the interface. Defaults to None.
            center_slab (bool): center the slab. Defaults to True.
        """
    interface_properties = interface_properties or {}
    if isinstance(substrate_slab, Slab):
        substrate_slab = substrate_slab.get_orthogonal_c_slab()
    if isinstance(film_slab, Slab):
        film_slab = film_slab.get_orthogonal_c_slab()
    assert_allclose(film_slab.lattice.alpha, 90, 0.1)
    assert_allclose(film_slab.lattice.beta, 90, 0.1)
    assert_allclose(substrate_slab.lattice.alpha, 90, 0.1)
    assert_allclose(substrate_slab.lattice.beta, 90, 0.1)
    sub_vecs = substrate_slab.lattice.matrix.copy()
    if np.dot(np.cross(*sub_vecs[:2]), sub_vecs[2]) < 0:
        sub_vecs[2] *= -1.0
        substrate_slab.lattice = Lattice(sub_vecs)
    sub_coords = substrate_slab.frac_coords
    film_coords = film_slab.frac_coords
    sub_min_c = np.min(sub_coords[:, 2]) * substrate_slab.lattice.c
    sub_max_c = np.max(sub_coords[:, 2]) * substrate_slab.lattice.c
    film_min_c = np.min(film_coords[:, 2]) * film_slab.lattice.c
    film_max_c = np.max(film_coords[:, 2]) * film_slab.lattice.c
    min_height = np.abs(film_max_c - film_min_c) + np.abs(sub_max_c - sub_min_c)
    abc = substrate_slab.lattice.abc[:2] + (min_height + gap + vacuum_over_film,)
    angles = substrate_slab.lattice.angles
    lattice = Lattice.from_parameters(*abc, *angles)
    species = substrate_slab.species + film_slab.species
    sub_coords = np.subtract(sub_coords, [0, 0, np.min(sub_coords[:, 2])])
    sub_coords[:, 2] *= substrate_slab.lattice.c / lattice.c
    film_coords[:, 2] *= -1.0
    film_coords[:, 2] *= film_slab.lattice.c / lattice.c
    film_coords = np.subtract(film_coords, [0, 0, np.min(film_coords[:, 2])])
    film_coords = np.add(film_coords, [0, 0, gap / lattice.c + np.max(sub_coords[:, 2])])
    coords = np.concatenate([sub_coords, film_coords])
    if center_slab:
        coords = np.add(coords, [0, 0, 0.5 - np.average(coords[:, 2])])
    site_properties = {}
    site_props_in_both = set(substrate_slab.site_properties) & set(film_slab.site_properties)
    for key in site_props_in_both:
        site_properties[key] = [*substrate_slab.site_properties[key], *film_slab.site_properties[key]]
    site_properties['interface_label'] = ['substrate'] * len(substrate_slab) + ['film'] * len(film_slab)
    iface = cls(lattice=lattice, species=species, coords=coords, to_unit_cell=False, coords_are_cartesian=False, site_properties=site_properties, validate_proximity=False, in_plane_offset=in_plane_offset, gap=gap, vacuum_over_film=vacuum_over_film, interface_properties=interface_properties)
    iface.sort()
    return iface