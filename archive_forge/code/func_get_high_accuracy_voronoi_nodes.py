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
def get_high_accuracy_voronoi_nodes(structure, rad_dict, probe_rad=0.1):
    """
    Analyze the void space in the input structure using high accuracy
    voronoi decomposition.
    Calls Zeo++ for Voronoi decomposition.

    Args:
        structure: pymatgen Structure
        rad_dict (optional): Dictionary of radii of elements in structure.
            If not given, Zeo++ default values are used.
            Note: Zeo++ uses atomic radii of elements.
            For ionic structures, pass rad_dict with ionic radii
        probe_rad (optional): Sampling probe radius in Angstroms.
            Default is 0.1 A

    Returns:
        voronoi nodes as pymatgen Structure within the
        unit cell defined by the lattice of input structure
        voronoi face centers as pymatgen Structure within the
        unit cell defined by the lattice of input structure
    """
    with ScratchDir('.'):
        name = 'temp_zeo1'
        zeo_inp_filename = f'{name}.cssr'
        ZeoCssr(structure).write_file(zeo_inp_filename)
        rad_flag = True
        rad_file = name + '.rad'
        with open(rad_file, 'w+') as file:
            for el in rad_dict:
                print(f'{el} {rad_dict[el].real}', file=file)
        atom_net = AtomNetwork.read_from_CSSR(zeo_inp_filename, rad_flag=rad_flag, rad_file=rad_file)
        red_ha_vornet = prune_voronoi_network_close_node(atom_net)
        red_ha_vornet.analyze_writeto_XYZ(name, probe_rad, atom_net)
        voro_out_filename = name + '_voro.xyz'
        voro_node_mol = ZeoVoronoiXYZ.from_file(voro_out_filename).molecule
    species = ['X'] * len(voro_node_mol)
    coords = []
    prop = []
    for site in voro_node_mol:
        coords.append(list(site.coords))
        prop.append(site.properties['voronoi_radius'])
    lattice = Lattice.from_parameters(*structure.lattice.parameters)
    return Structure(lattice, species, coords, coords_are_cartesian=True, to_unit_cell=True, site_properties={'voronoi_radius': prop})