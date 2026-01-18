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
def parse_omx_version(txt):
    """Parse version number from stdout header."""
    match = re.search('Welcome to OpenMX\\s+Ver\\.\\s+(\\S+)', txt, re.M)
    return match.group(1)