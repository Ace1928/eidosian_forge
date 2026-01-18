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
def get_free_sphere_params(structure, rad_dict=None, probe_rad=0.1):
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
        voronoi nodes as pymatgen Structure within the
        unit cell defined by the lattice of input structure
        voronoi face centers as pymatgen Structure within the
        unit cell defined by the lattice of input structure
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
        out_file = 'temp.res'
        atom_net.calculate_free_sphere_parameters(out_file)
        if os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
            with open(out_file) as file:
                output = file.readline()
        else:
            output = ''
    fields = [val.strip() for val in output.split()][1:4]
    if len(fields) == 3:
        fields = [float(field) for field in fields]
        free_sphere_params = {'inc_sph_max_dia': fields[0], 'free_sph_max_dia': fields[1], 'inc_sph_along_free_sph_path_max_dia': fields[2]}
    return free_sphere_params