import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert
def split_contents_by_section(raw_datafile_contents):
    return re.split('^([A-Za-z]+\\s*)$\\n', raw_datafile_contents, flags=re.MULTILINE)