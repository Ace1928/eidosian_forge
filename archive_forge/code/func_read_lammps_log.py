import os
import shutil
import shlex
from subprocess import Popen, PIPE, TimeoutExpired
from threading import Thread
from re import compile as re_compile, IGNORECASE
from tempfile import mkdtemp, NamedTemporaryFile, mktemp as uns_mktemp
import inspect
import warnings
from typing import Dict, Any
import numpy as np
from ase import Atoms
from ase.parallel import paropen
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from ase.data import chemical_symbols
from ase.data import atomic_masses
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump
from ase.calculators.lammps import Prism
from ase.calculators.lammps import write_lammps_in
from ase.calculators.lammps import CALCULATION_END_MARK
from ase.calculators.lammps import convert
def read_lammps_log(self, lammps_log=None):
    """Method which reads a LAMMPS output log file."""
    if lammps_log is None:
        lammps_log = self.label + '.log'
    if isinstance(lammps_log, str):
        fileobj = paropen(lammps_log, 'wb')
        close_log_file = True
    else:
        fileobj = lammps_log
        close_log_file = False
    _custom_thermo_mark = ' '.join([x.capitalize() for x in self.parameters.thermo_args[0:3]])
    f_re = '([+-]?(?:(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:e[+-]?\\d+)?|nan|inf))'
    n_args = len(self.parameters['thermo_args'])
    _custom_thermo_re = re_compile('^\\s*' + '\\s+'.join([f_re] * n_args) + '\\s*$', flags=IGNORECASE)
    thermo_content = []
    line = fileobj.readline().decode('utf-8')
    while line and line.strip() != CALCULATION_END_MARK:
        if 'ERROR:' in line:
            if close_log_file:
                fileobj.close()
            raise RuntimeError(f'LAMMPS exits with error message: {line}')
        if line.startswith(_custom_thermo_mark):
            bool_match = True
            while bool_match:
                line = fileobj.readline().decode('utf-8')
                bool_match = _custom_thermo_re.match(line)
                if bool_match:
                    thermo_content.append(dict(zip(self.parameters.thermo_args, map(float, bool_match.groups()))))
        else:
            line = fileobj.readline().decode('utf-8')
    if close_log_file:
        fileobj.close()
    self.thermo_content = thermo_content