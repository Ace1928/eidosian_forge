import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def generate_gromacs_run_file(self):
    """ Generates input file for a gromacs mdrun
        based on structure file and topology file
        resulting file is self.label + '.tpr
        """
    try:
        os.remove(self.label + '.tpr')
    except OSError:
        pass
    subcmd = 'grompp'
    command = ' '.join([subcmd, '-f', self.label + '.mdp', '-c', self.label + '.g96', '-p', self.label + '.top', '-o', self.label + '.tpr', '-maxwarn', '100', self.params_runs.get('extra_grompp_parameters', ''), '> {}.{}.log 2>&1'.format(self.label, subcmd)])
    self._execute_gromacs(command)