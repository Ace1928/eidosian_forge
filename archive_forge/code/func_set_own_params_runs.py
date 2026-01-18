import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def set_own_params_runs(self, key, value):
    """Set own gromacs parameter for program parameters
        Add spaces to avoid errors """
    self.params_runs[key] = ' ' + value + ' '