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
def read_parameters(self):
    """read parameters from control file"""

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
    params = {}
    pdgs = {}
    for p in self.parameter_group:
        if self.parameter_group[p] and self.parameter_key[p]:
            pdgs[p] = parse_data_group(read_data_group(self.parameter_group[p]), self.parameter_group[p])
    for p in self.parameter_key:
        if self.parameter_key[p]:
            if self.parameter_key[p] == self.parameter_group[p]:
                if pdgs[p] is None:
                    if self.parameter_type[p] is bool:
                        params[p] = False
                    else:
                        params[p] = None
                elif self.parameter_type[p] is bool:
                    params[p] = True
                else:
                    typ = self.parameter_type[p]
                    val = typ(pdgs[p])
                    mapping = self.parameter_mapping
                    if p in list(mapping.keys()):
                        fun = mapping[p]['from_control']
                        val = fun(val)
                    params[p] = val
            elif pdgs[p] is None:
                params[p] = None
            elif isinstance(pdgs[p], str):
                if self.parameter_type[p] is bool:
                    params[p] = pdgs[p] == self.parameter_key[p]
            elif self.parameter_key[p] not in list(pdgs[p].keys()):
                if self.parameter_type[p] is bool:
                    params[p] = False
                else:
                    params[p] = None
            else:
                typ = self.parameter_type[p]
                val = typ(pdgs[p][self.parameter_key[p]])
                mapping = self.parameter_mapping
                if p in list(mapping.keys()):
                    fun = mapping[p]['from_control']
                    val = fun(val)
                params[p] = val
    basis_sets = set([bs['nickname'] for bs in self.results['basis set']])
    assert len(basis_sets) == 1
    params['basis set name'] = list(basis_sets)[0]
    params['basis set definition'] = self.results['basis set']
    orbs = self.results['molecular orbitals']
    params['rohf'] = bool(len(read_data_group('rohf'))) or bool(len(read_data_group('roothaan')))
    core_charge = 0
    if self.results['ecps']:
        for ecp in self.results['ecps']:
            for symbol in self.atoms.get_chemical_symbols():
                if symbol.lower() == ecp['element'].lower():
                    core_charge -= ecp['number of core electrons']
    if params['uhf']:
        alpha_occ = [o['occupancy'] for o in orbs if o['spin'] == 'alpha']
        beta_occ = [o['occupancy'] for o in orbs if o['spin'] == 'beta']
        spin = (np.sum(alpha_occ) - np.sum(beta_occ)) * 0.5
        params['multiplicity'] = int(2 * spin + 1)
        nuclear_charge = np.sum(self.atoms.numbers)
        electron_charge = -int(np.sum(alpha_occ) + np.sum(beta_occ))
        electron_charge += core_charge
        params['total charge'] = nuclear_charge + electron_charge
    elif not params['rohf']:
        params['multiplicity'] = 1
        nuclear_charge = np.sum(self.atoms.numbers)
        electron_charge = -int(np.sum([o['occupancy'] for o in orbs]))
        electron_charge += core_charge
        params['total charge'] = nuclear_charge + electron_charge
    else:
        raise NotImplementedError('ROHF not implemented')
    if os.path.exists('job.start'):
        with open('job.start', 'r') as log:
            lines = log.readlines()
        for line in lines:
            if 'CRITERION FOR TOTAL SCF-ENERGY' in line:
                en = int(re.search('10\\*{2}\\(-(\\d+)\\)', line).group(1))
                params['energy convergence'] = en
            if 'CRITERION FOR MAXIMUM NORM OF SCF-ENERGY GRADIENT' in line:
                gr = int(re.search('10\\*{2}\\(-(\\d+)\\)', line).group(1))
                params['force convergence'] = gr
            if 'AN OPTIMIZATION WITH MAX' in line:
                cy = int(re.search('MAX. (\\d+) CYCLES', line).group(1))
                params['geometry optimization iterations'] = cy
    return params