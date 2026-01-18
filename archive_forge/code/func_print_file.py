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
def print_file(self, file=None, running=None, **args):
    """ Print the file while calculation is running"""
    prev_position = 0
    last_position = 0
    while not os.path.isfile(file):
        self.prind('Waiting for %s to come out' % file)
        time.sleep(5)
    with open(file, 'r') as fd:
        while running(**args):
            fd.seek(last_position)
            new_data = fd.read()
            prev_position = fd.tell()
            if prev_position != last_position:
                if not self.nohup:
                    print(new_data)
                last_position = prev_position
            time.sleep(1)