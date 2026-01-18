import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def generate_topology_and_g96file(self):
    """ from coordinates (self.label.+'pdb')
            and gromacs run input file (self.label + '.mdp)
            generate topology (self.label+'top')
            and structure file in .g96 format (self.label + '.g96')
        """
    subcmd = 'pdb2gmx'
    command = ' '.join([subcmd, '-f', self.params_runs['init_structure'], '-o', self.label + '.g96', '-p', self.label + '.top', '-ff', self.params_runs['force_field'], '-water', self.params_runs['water'], self.params_runs.get('extra_pdb2gmx_parameters', ''), '> {}.{}.log 2>&1'.format(self.label, subcmd)])
    self._execute_gromacs(command)
    atoms = read_gromos(self.label + '.g96')
    self.atoms = atoms.copy()