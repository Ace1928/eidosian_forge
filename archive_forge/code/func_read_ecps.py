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
def read_ecps(self):
    """read the effective core potentials"""
    ecpf = read_data_group('ecp')
    if not bool(len(ecpf)):
        self.results['ecps'] = None
        self.results['ecps formatted'] = None
        return
    self.results['ecps'] = []
    self.results['ecps formatted'] = {}
    self.results['ecps formatted']['turbomole'] = ecpf
    lines = ecpf.split('\n')
    ecp = {}
    groups = []
    group = {}
    terms = []
    read_tag = False
    read_data = False
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if '$ecp' in line:
            continue
        if '$end' in line:
            break
        if re.match('^\\s*#', line):
            continue
        if re.match('^\\s*\\*', line):
            if read_tag:
                read_tag = False
                read_data = True
            else:
                if read_data:
                    group['terms'] = terms
                    group['number of terms'] = len(terms)
                    terms = []
                    groups.append(group)
                    group = {}
                    ecp['groups'] = groups
                    groups = []
                    self.results['ecps'].append(ecp)
                    ecp = {}
                    read_data = False
                read_tag = True
            continue
        if read_tag:
            match = re.search('^\\s*(\\w+)\\s+(.+)', line)
            if match:
                ecp['element'] = match.group(1)
                ecp['nickname'] = match.group(2)
            else:
                raise RuntimeError('error reading ecp')
        else:
            regex = 'ncore\\s*=\\s*(\\d+)\\s+lmax\\s*=\\s*(\\d+)'
            match = re.search(regex, line)
            if match:
                ecp['number of core electrons'] = int(match.group(1))
                ecp['maximum angular momentum number'] = int(match.group(2))
                continue
            match = re.search('^(\\w(\\-\\w)?)', line)
            if match:
                if len(terms) > 0:
                    group['terms'] = terms
                    group['number of terms'] = len(terms)
                    terms = []
                    groups.append(group)
                    group = {}
                group['title'] = str(match.group(1))
                continue
            regex = '^\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)\\s+(\\d)\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'
            match = re.search(regex, line)
            if match:
                terms.append({'coefficient': float(match.group(1)), 'power of r': float(match.group(3)), 'exponent': float(match.group(4))})