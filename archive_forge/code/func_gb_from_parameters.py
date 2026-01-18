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
def gb_from_parameters(self, rotation_axis, rotation_angle, expand_times=4, vacuum_thickness=0.0, ab_shift: tuple[float, float]=(0, 0), normal=False, ratio=None, plane=None, max_search=20, tol_coi=1e-08, rm_ratio=0.7, quick_gen=False):
    """
        Args:
            rotation_axis (list): Rotation axis of GB in the form of a list of integer
                e.g.: [1, 1, 0]
            rotation_angle (float, in unit of degree): rotation angle used to generate GB.
                Make sure the angle is accurate enough. You can use the enum* functions
                in this class to extract the accurate angle.
                e.g.: The rotation angle of sigma 3 twist GB with the rotation axis
                [1, 1, 1] and GB plane (1, 1, 1) can be 60 degree.
                If you do not know the rotation angle, but know the sigma value, we have
                provide the function get_rotation_angle_from_sigma which is able to return
                all the rotation angles of sigma value you provided.
            expand_times (int): The multiple times used to expand one unit grain to larger grain.
                This is used to tune the grain length of GB to warrant that the two GBs in one
                cell do not interact with each other. Default set to 4.
            vacuum_thickness (float, in angstrom): The thickness of vacuum that you want to insert
                between two grains of the GB. Default to 0.
            ab_shift (list of float, in unit of a, b vectors of Gb): in plane shift of two grains
            normal (logic):
                determine if need to require the c axis of top grain (first transformation matrix)
                perpendicular to the surface or not.
                default to false.
            ratio (list of integers):
                lattice axial ratio.
                For cubic system, ratio is not needed.
                For tetragonal system, ratio = [mu, mv], list of two integers,
                that is, mu/mv = c2/a2. If it is irrational, set it to none.
                For orthorhombic system, ratio = [mu, lam, mv], list of 3 integers,
                that is, mu:lam:mv = c2:b2:a2. If irrational for one axis, set it to None.
                e.g. mu:lam:mv = c2,None,a2, means b2 is irrational.
                For rhombohedral system, ratio = [mu, mv], list of two integers,
                that is, mu/mv is the ratio of (1+2*cos(alpha))/cos(alpha).
                If irrational, set it to None.
                For hexagonal system, ratio = [mu, mv], list of two integers,
                that is, mu/mv = c2/a2. If it is irrational, set it to none.
                This code also supplies a class method to generate the ratio from the
                structure (get_ratio). User can also make their own approximation and
                input the ratio directly.
            plane (list): Grain boundary plane in the form of a list of integers
                e.g.: [1, 2, 3]. If none, we set it as twist GB. The plane will be perpendicular
                to the rotation axis.
            max_search (int): max search for the GB lattice vectors that give the smallest GB
                lattice. If normal is true, also max search the GB c vector that perpendicular
                to the plane. For complex GB, if you want to speed up, you can reduce this value.
                But too small of this value may lead to error.
            tol_coi (float): tolerance to find the coincidence sites. When making approximations to
                the ratio needed to generate the GB, you probably need to increase this tolerance to
                obtain the correct number of coincidence sites. To check the number of coincidence
                sites are correct or not, you can compare the generated Gb object's sigma_from_site_prop
                with enum* sigma values (what user expected by input).
            rm_ratio (float): the criteria to remove the atoms which are too close with each other.
                rm_ratio*bond_length of bulk system is the criteria of bond length, below which the atom
                will be removed. Default to 0.7.
            quick_gen (bool): whether to quickly generate a supercell, if set to true, no need to
                find the smallest cell.

        Returns:
            Grain boundary structure (GB object).
        """
    lat_type = self.lat_type
    trans_cry = np.eye(3)
    if lat_type == 'c':
        analyzer = SpacegroupAnalyzer(self.initial_structure)
        convention_cell = analyzer.get_conventional_standard_structure()
        vol_ratio = self.initial_structure.volume / convention_cell.volume
        if abs(vol_ratio - 0.5) < 0.001:
            trans_cry = np.array([[0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]])
            logger.info('Make sure this is for cubic with bcc primitive cell')
        elif abs(vol_ratio - 0.25) < 0.001:
            trans_cry = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
            logger.info('Make sure this is for cubic with fcc primitive cell')
        else:
            logger.info('Make sure this is for cubic with conventional cell')
    elif lat_type == 't':
        logger.info('Make sure this is for tetragonal system')
        if ratio is None:
            logger.info('Make sure this is for irrational c2/a2')
        elif len(ratio) != 2:
            raise RuntimeError('Tetragonal system needs correct c2/a2 ratio')
    elif lat_type == 'o':
        logger.info('Make sure this is for orthorhombic system')
        if ratio is None:
            raise RuntimeError('CSL does not exist if all axial ratios are irrational for an orthorhombic system')
        if len(ratio) != 3:
            raise RuntimeError('Orthorhombic system needs correct c2:b2:a2 ratio')
    elif lat_type == 'h':
        logger.info('Make sure this is for hexagonal system')
        if ratio is None:
            logger.info('Make sure this is for irrational c2/a2')
        elif len(ratio) != 2:
            raise RuntimeError('Hexagonal system needs correct c2/a2 ratio')
    elif lat_type == 'r':
        logger.info('Make sure this is for rhombohedral system')
        if ratio is None:
            logger.info('Make sure this is for irrational (1+2*cos(alpha)/cos(alpha) ratio')
        elif len(ratio) != 2:
            raise RuntimeError('Rhombohedral system needs correct (1+2*cos(alpha)/cos(alpha) ratio')
    else:
        raise RuntimeError('Lattice type not implemented. This code works for cubic, tetragonal, orthorhombic, rhombohedral, hexagonal systems')
    if len(rotation_axis) == 4:
        u1 = rotation_axis[0]
        v1 = rotation_axis[1]
        w1 = rotation_axis[3]
        if lat_type.lower() == 'h':
            u = 2 * u1 + v1
            v = 2 * v1 + u1
            w = w1
            rotation_axis = [u, v, w]
        elif lat_type.lower() == 'r':
            u = 2 * u1 + v1 + w1
            v = v1 + w1 - u1
            w = w1 - 2 * v1 - u1
            rotation_axis = [u, v, w]
    if reduce(gcd, rotation_axis) != 1:
        rotation_axis = [int(round(x / reduce(gcd, rotation_axis))) for x in rotation_axis]
    if plane is not None and len(plane) == 4:
        u1, v1, w1 = (plane[0], plane[1], plane[3])
        plane = [u1, v1, w1]
    if plane is None:
        if lat_type.lower() == 'c':
            plane = rotation_axis
        else:
            if lat_type.lower() == 'h':
                c2_a2_ratio = 1.0 if ratio is None else ratio[0] / ratio[1]
                metric = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, c2_a2_ratio]])
            elif lat_type.lower() == 'r':
                cos_alpha = 0.5 if ratio is None else 1.0 / (ratio[0] / ratio[1] - 2)
                metric = np.array([[1, cos_alpha, cos_alpha], [cos_alpha, 1, cos_alpha], [cos_alpha, cos_alpha, 1]])
            elif lat_type.lower() == 't':
                c2_a2_ratio = 1.0 if ratio is None else ratio[0] / ratio[1]
                metric = np.array([[1, 0, 0], [0, 1, 0], [0, 0, c2_a2_ratio]])
            elif lat_type.lower() == 'o':
                for idx in range(3):
                    if ratio[idx] is None:
                        ratio[idx] = 1
                metric = np.array([[1, 0, 0], [0, ratio[1] / ratio[2], 0], [0, 0, ratio[0] / ratio[2]]])
            else:
                raise RuntimeError('Lattice type has not implemented.')
            plane = np.matmul(rotation_axis, metric)
            fractions = [Fraction(x).limit_denominator() for x in plane]
            least_mul = reduce(lcm, [fraction.denominator for fraction in fractions])
            plane = [int(round(x * least_mul)) for x in plane]
    if reduce(gcd, plane) != 1:
        index = reduce(gcd, plane)
        plane = [int(round(x / index)) for x in plane]
    t1, t2 = self.get_trans_mat(r_axis=rotation_axis, angle=rotation_angle, normal=normal, trans_cry=trans_cry, lat_type=lat_type, ratio=ratio, surface=plane, max_search=max_search, quick_gen=quick_gen)
    if lat_type.lower() != 'c':
        if lat_type.lower() == 'h':
            if ratio is None:
                mu, mv = [1, 1]
            else:
                mu, mv = ratio
            trans_cry1 = np.array([[1, 0, 0], [-0.5, np.sqrt(3.0) / 2.0, 0], [0, 0, np.sqrt(mu / mv)]])
        elif lat_type.lower() == 'r':
            if ratio is None:
                c2_a2_ratio = 1.0
            else:
                mu, mv = ratio
                c2_a2_ratio = 3 / (2 - 6 * mv / mu)
            trans_cry1 = np.array([[0.5, np.sqrt(3.0) / 6.0, 1.0 / 3 * np.sqrt(c2_a2_ratio)], [-0.5, np.sqrt(3.0) / 6.0, 1.0 / 3 * np.sqrt(c2_a2_ratio)], [0, -1 * np.sqrt(3.0) / 3.0, 1.0 / 3 * np.sqrt(c2_a2_ratio)]])
        else:
            if lat_type.lower() == 't':
                if ratio is None:
                    mu, mv = [1, 1]
                else:
                    mu, mv = ratio
                lam = mv
            elif lat_type.lower() == 'o':
                new_ratio = [1 if v is None else v for v in ratio]
                mu, lam, mv = new_ratio
            trans_cry1 = np.array([[1, 0, 0], [0, np.sqrt(lam / mv), 0], [0, 0, np.sqrt(mu / mv)]])
    else:
        trans_cry1 = trans_cry
    grain_matrix = np.dot(t2, trans_cry1)
    plane_init = np.cross(grain_matrix[0], grain_matrix[1])
    if lat_type.lower() != 'c':
        plane_init = np.dot(plane_init, trans_cry1.T)
    join_plane = self.vec_to_surface(plane_init)
    parent_structure = self.initial_structure.copy()
    if len(parent_structure) == 1:
        temp_str = parent_structure.copy()
        temp_str.make_supercell([1, 1, 2])
        distance = temp_str.distance_matrix
    else:
        distance = parent_structure.distance_matrix
    bond_length = np.min(distance[np.nonzero(distance)])
    top_grain = fix_pbc(parent_structure * t1)
    if normal and (not quick_gen):
        t_temp = self.get_trans_mat(r_axis=rotation_axis, angle=rotation_angle, normal=False, trans_cry=trans_cry, lat_type=lat_type, ratio=ratio, surface=plane, max_search=max_search)
        oriented_unit_cell = fix_pbc(parent_structure * t_temp[0])
        t_matrix = oriented_unit_cell.lattice.matrix
        normal_v_plane = np.cross(t_matrix[0], t_matrix[1])
        unit_normal_v = normal_v_plane / np.linalg.norm(normal_v_plane)
        unit_ab_adjust = (t_matrix[2] - np.dot(unit_normal_v, t_matrix[2]) * unit_normal_v) / np.dot(unit_normal_v, t_matrix[2])
    else:
        oriented_unit_cell = top_grain.copy()
        unit_ab_adjust = 0.0
    bottom_grain = fix_pbc(parent_structure * t2, top_grain.lattice.matrix)
    n_sites = len(top_grain)
    t_and_b = Structure(top_grain.lattice, top_grain.species + bottom_grain.species, list(top_grain.frac_coords) + list(bottom_grain.frac_coords))
    t_and_b_dis = t_and_b.lattice.get_all_distances(t_and_b.frac_coords[0:n_sites], t_and_b.frac_coords[n_sites:n_sites * 2])
    index_incident = np.nonzero(t_and_b_dis < np.min(t_and_b_dis) + tol_coi)
    top_labels = []
    for idx in range(n_sites):
        if idx in index_incident[0]:
            top_labels.append('top_incident')
        else:
            top_labels.append('top')
    bottom_labels = []
    for idx in range(n_sites):
        if idx in index_incident[1]:
            bottom_labels.append('bottom_incident')
        else:
            bottom_labels.append('bottom')
    top_grain = Structure(Lattice(top_grain.lattice.matrix), top_grain.species, top_grain.frac_coords, site_properties={'grain_label': top_labels})
    bottom_grain = Structure(Lattice(bottom_grain.lattice.matrix), bottom_grain.species, bottom_grain.frac_coords, site_properties={'grain_label': bottom_labels})
    top_grain.make_supercell([1, 1, expand_times])
    bottom_grain.make_supercell([1, 1, expand_times])
    top_grain = fix_pbc(top_grain)
    bottom_grain = fix_pbc(bottom_grain)
    edge_b = 1.0 - max(bottom_grain.frac_coords[:, 2])
    edge_t = 1.0 - max(top_grain.frac_coords[:, 2])
    c_adjust = (edge_t - edge_b) / 2.0
    all_species = []
    all_species.extend([site.specie for site in bottom_grain])
    all_species.extend([site.specie for site in top_grain])
    half_lattice = top_grain.lattice
    normal_v_plane = np.cross(half_lattice.matrix[0], half_lattice.matrix[1])
    unit_normal_v = normal_v_plane / np.linalg.norm(normal_v_plane)
    translation_v = unit_normal_v * vacuum_thickness
    whole_matrix_no_vac = np.array(half_lattice.matrix)
    whole_matrix_no_vac[2] = half_lattice.matrix[2] * 2
    whole_matrix_with_vac = whole_matrix_no_vac.copy()
    whole_matrix_with_vac[2] = whole_matrix_no_vac[2] + translation_v * 2
    whole_lat = Lattice(whole_matrix_with_vac)
    all_coords = []
    grain_labels = bottom_grain.site_properties['grain_label'] + top_grain.site_properties['grain_label']
    for site in bottom_grain:
        all_coords.append(site.coords)
    for site in top_grain:
        all_coords.append(site.coords + half_lattice.matrix[2] * (1 + c_adjust) + unit_ab_adjust * np.linalg.norm(half_lattice.matrix[2] * (1 + c_adjust)) + translation_v + ab_shift[0] * whole_matrix_with_vac[0] + ab_shift[1] * whole_matrix_with_vac[1])
    gb_with_vac = Structure(whole_lat, all_species, all_coords, coords_are_cartesian=True, site_properties={'grain_label': grain_labels})
    cos_c_norm_plane = np.dot(unit_normal_v, whole_matrix_with_vac[2]) / whole_lat.c
    range_c_len = abs(bond_length / cos_c_norm_plane / whole_lat.c)
    sites_near_gb = []
    sites_away_gb: list[PeriodicSite] = []
    for site in gb_with_vac:
        if site.frac_coords[2] < range_c_len or site.frac_coords[2] > 1 - range_c_len or (site.frac_coords[2] > 0.5 - range_c_len and site.frac_coords[2] < 0.5 + range_c_len):
            sites_near_gb.append(site)
        else:
            sites_away_gb.append(site)
    if len(sites_near_gb) >= 1:
        s_near_gb = Structure.from_sites(sites_near_gb)
        s_near_gb.merge_sites(tol=bond_length * rm_ratio, mode='d')
        all_sites = sites_away_gb + s_near_gb.sites
        gb_with_vac = Structure.from_sites(all_sites)
    gb_with_vac = fix_pbc(gb_with_vac, whole_lat.matrix)
    return GrainBoundary(whole_lat, gb_with_vac.species, gb_with_vac.cart_coords, rotation_axis, rotation_angle, plane, join_plane, self.initial_structure, vacuum_thickness, ab_shift, site_properties=gb_with_vac.site_properties, oriented_unit_cell=oriented_unit_cell, coords_are_cartesian=True)