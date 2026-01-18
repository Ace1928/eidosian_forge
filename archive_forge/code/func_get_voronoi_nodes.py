from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.dev import requires
from monty.io import zopen
from monty.tempfile import ScratchDir
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.cssr import Cssr
from pymatgen.io.xyz import XYZ
@requires(zeo_found, 'get_voronoi_nodes requires Zeo++ cython extension to be installed. Please contact developers of Zeo++ to obtain it.')
def get_voronoi_nodes(structure, rad_dict=None, probe_rad=0.1):
    """
    Analyze the void space in the input structure using voronoi decomposition
    Calls Zeo++ for Voronoi decomposition.

    Args:
        structure: pymatgen Structure
        rad_dict (optional): Dictionary of radii of elements in structure.
            If not given, Zeo++ default values are used.
            Note: Zeo++ uses atomic radii of elements.
            For ionic structures, pass rad_dict with ionic radii
        probe_rad (optional): Sampling probe radius in Angstroms. Default is
            0.1 A

    Returns:
        voronoi nodes as pymatgen Structure within the unit cell defined by the lattice of
        input structure voronoi face centers as pymatgen Structure within the unit cell
        defined by the lattice of input structure
    """
    with ScratchDir('.'):
        name = 'temp_zeo1'
        zeo_inp_filename = name + '.cssr'
        ZeoCssr(structure).write_file(zeo_inp_filename)
        rad_file = None
        rad_flag = False
        if rad_dict:
            rad_file = name + '.rad'
            rad_flag = True
            with open(rad_file, 'w+') as file:
                for el in rad_dict:
                    file.write(f'{el} {rad_dict[el].real}\n')
        atom_net = AtomNetwork.read_from_CSSR(zeo_inp_filename, rad_flag=rad_flag, rad_file=rad_file)
        vor_net, vor_edge_centers, vor_face_centers = atom_net.perform_voronoi_decomposition()
        vor_net.analyze_writeto_XYZ(name, probe_rad, atom_net)
        voro_out_filename = name + '_voro.xyz'
        voro_node_mol = ZeoVoronoiXYZ.from_file(voro_out_filename).molecule
    species = ['X'] * len(voro_node_mol)
    coords = []
    prop = []
    for site in voro_node_mol:
        coords.append(list(site.coords))
        prop.append(site.properties['voronoi_radius'])
    lattice = Lattice.from_parameters(*structure.lattice.parameters)
    vor_node_struct = Structure(lattice, species, coords, coords_are_cartesian=True, to_unit_cell=True, site_properties={'voronoi_radius': prop})
    rot_face_centers = [(center[1], center[2], center[0]) for center in vor_face_centers]
    rot_edge_centers = [(center[1], center[2], center[0]) for center in vor_edge_centers]
    species = ['X'] * len(rot_face_centers)
    prop = [0.0] * len(rot_face_centers)
    vor_facecenter_struct = Structure(lattice, species, rot_face_centers, coords_are_cartesian=True, to_unit_cell=True, site_properties={'voronoi_radius': prop})
    species = ['X'] * len(rot_edge_centers)
    prop = [0.0] * len(rot_edge_centers)
    vor_edgecenter_struct = Structure(lattice, species, rot_edge_centers, coords_are_cartesian=True, to_unit_cell=True, site_properties={'voronoi_radius': prop})
    return (vor_node_struct, vor_edgecenter_struct, vor_facecenter_struct)