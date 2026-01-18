import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def read_ldau(self, lines=None):
    """Read the LDA+U values from OUTCAR"""
    if not lines:
        lines = self.load_file('OUTCAR')
    ldau_luj = None
    ldauprint = None
    ldau = None
    ldautype = None
    atomtypes = []
    for line in lines:
        if line.find('TITEL') != -1:
            atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
        if line.find('LDAUTYPE') != -1:
            ldautype = int(line.split('=')[-1])
            ldau = True
            ldau_luj = {}
        if line.find('LDAUL') != -1:
            L = line.split('=')[-1].split()
        if line.find('LDAUU') != -1:
            U = line.split('=')[-1].split()
        if line.find('LDAUJ') != -1:
            J = line.split('=')[-1].split()
    if ldau:
        for i, symbol in enumerate(atomtypes):
            ldau_luj[symbol] = {'L': int(L[i]), 'U': float(U[i]), 'J': float(J[i])}
        self.dict_params['ldau_luj'] = ldau_luj
    self.ldau = ldau
    self.ldauprint = ldauprint
    self.ldautype = ldautype
    self.ldau_luj = ldau_luj
    return (ldau, ldauprint, ldautype, ldau_luj)