import pytest
import numpy as np
from ase import Atom
from ase.build import bulk
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet
@pytest.fixture
def calc_params_Fe(lammps_data_file_Fe):
    calc_params = {}
    calc_params['lammps_header'] = ['units           real', 'atom_style      full', 'boundary        p p p', 'box tilt        large', 'pair_style      lj/cut/coul/long 12.500', 'bond_style      harmonic', 'angle_style     harmonic', 'kspace_style    ewald 0.0001', 'kspace_modify   gewald 0.01', f'read_data      {lammps_data_file_Fe}']
    calc_params['lmpcmds'] = []
    calc_params['atom_types'] = {'Fe': 1}
    calc_params['create_atoms'] = False
    calc_params['create_box'] = False
    calc_params['boundary'] = False
    calc_params['log_file'] = 'test.log'
    calc_params['keep_alive'] = True
    return calc_params