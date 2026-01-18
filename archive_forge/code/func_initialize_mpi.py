import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def initialize_mpi(self, mpi):
    if mpi:
        self.mpi = dict(self.default_mpi)
        for key in mpi:
            if key not in self.default_mpi:
                allowed = ', '.join(list(self.default_mpi.keys()))
                raise TypeError('Unexpected keyword "{0}" in "mpi" dictionary.  Must be one of: {1}'.format(key, allowed))
        self.mpi.update(mpi)
        self.__dict__.update(self.mpi)
    else:
        self.mpi = None