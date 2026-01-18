import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def parse_data_group(dg, dg_name):
    """parse a data group"""
    if len(dg) == 0:
        return None
    lsep = None
    ksep = None
    ndg = dg.replace('$' + dg_name, '').strip()
    if '\n' in ndg:
        lsep = '\n'
    if '=' in ndg:
        ksep = '='
    if not lsep and (not ksep):
        return ndg
    result = {}
    lines = ndg.split(lsep)
    for line in lines:
        fields = line.strip().split(ksep)
        if len(fields) == 2:
            result[fields[0]] = fields[1]
        elif len(fields) == 1:
            result[fields[0]] = True
    return result