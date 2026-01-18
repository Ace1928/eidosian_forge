import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def run_editconf(self):
    """ run gromacs program editconf, typically to set a simulation box
        writing to the input structure"""
    subcmd = 'editconf'
    command = ' '.join([subcmd, '-f', self.label + '.g96', '-o', self.label + '.g96', self.params_runs.get('extra_editconf_parameters', ''), '> {}.{}.log 2>&1'.format(self.label, subcmd)])
    self._execute_gromacs(command)