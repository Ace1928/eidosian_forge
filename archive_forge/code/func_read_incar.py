import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def read_incar(self, filename):
    """Method that imports settings from INCAR file.

        Typically named INCAR."""
    self.spinpol = False
    with open(filename, 'r') as fd:
        lines = fd.readlines()
    for line in lines:
        try:
            line = line.replace('*', ' * ')
            line = line.replace('=', ' = ')
            line = line.replace('#', '# ')
            data = line.split()
            if len(data) == 0:
                continue
            elif data[0][0] in ['#', '!']:
                continue
            key = data[0].lower()
            if '<Custom ASE key>' in line:
                value = line.split('=', 1)[1]
                value = value.split('#', 1)[0].strip()
                self.input_params['custom'][key] = value
            elif key in float_keys:
                self.float_params[key] = float(data[2])
            elif key in exp_keys:
                self.exp_params[key] = float(data[2])
            elif key in string_keys:
                self.string_params[key] = str(data[2])
            elif key in int_keys:
                if key == 'ispin':
                    self.int_params[key] = int(data[2])
                    if int(data[2]) == 2:
                        self.spinpol = True
                else:
                    self.int_params[key] = int(data[2])
            elif key in bool_keys:
                if 'true' in data[2].lower():
                    self.bool_params[key] = True
                elif 'false' in data[2].lower():
                    self.bool_params[key] = False
            elif key in list_bool_keys:
                self.list_bool_params[key] = [_from_vasp_bool(x) for x in _args_without_comment(data[2:])]
            elif key in list_int_keys:
                self.list_int_params[key] = [int(x) for x in _args_without_comment(data[2:])]
            elif key in list_float_keys:
                if key == 'magmom':
                    lst = []
                    i = 2
                    while i < len(data):
                        if data[i] in ['#', '!']:
                            break
                        if data[i] == '*':
                            b = lst.pop()
                            i += 1
                            for j in range(int(b)):
                                lst.append(float(data[i]))
                        else:
                            lst.append(float(data[i]))
                        i += 1
                    self.list_float_params['magmom'] = lst
                    lst = np.array(lst)
                    if self.atoms is not None:
                        self.atoms.set_initial_magnetic_moments(lst[self.resort])
                else:
                    data = _args_without_comment(data)
                    self.list_float_params[key] = [float(x) for x in data[2:]]
            elif key in special_keys:
                if key == 'lreal':
                    if 'true' in data[2].lower():
                        self.special_params[key] = True
                    elif 'false' in data[2].lower():
                        self.special_params[key] = False
                    else:
                        self.special_params[key] = data[2]
        except KeyError:
            raise IOError('Keyword "%s" in INCAR isnot known by calculator.' % key)
        except IndexError:
            raise IOError('Value missing for keyword "%s".' % key)