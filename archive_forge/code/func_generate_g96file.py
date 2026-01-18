import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def generate_g96file(self):
    """ from current coordinates (self.structure_file)
            write a structure file in .g96 format
        """
    write_gromos(self.label + '.g96', self.atoms)