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
def initialize_pbs(self, pbs):
    if pbs:
        self.pbs = dict(self.default_pbs)
        for key in pbs:
            if key not in self.default_pbs:
                allowed = ', '.join(list(self.default_pbs.keys()))
                raise TypeError('Unexpected keyword "{0}" in "pbs" dictionary.  Must be one of: {1}'.format(key, allowed))
        self.pbs.update(pbs)
        self.__dict__.update(self.pbs)
    else:
        self.pbs = None