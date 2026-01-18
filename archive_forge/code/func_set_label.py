import os
import os.path as op
import subprocess
import shutil
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray
def set_label(self, label):
    """Set label directory """
    self.label = label
    self.directory = self.label
    if self.directory == '':
        self.directory = os.curdir