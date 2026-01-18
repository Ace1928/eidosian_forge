import os
from subprocess import Popen, PIPE
import re
import numpy as np
from ase.units import Hartree, Bohr
from ase.calculators.calculator import PropertyNotImplementedError
def run_executable(self, mode='fleur', executable='FLEUR'):
    assert executable in ['FLEUR', 'FLEUR_SERIAL']
    executable_use = executable
    if executable == 'FLEUR_SERIAL' and (not os.environ.get(executable, '')):
        executable_use = 'FLEUR'
    try:
        code_exe = os.environ[executable_use]
    except KeyError:
        raise RuntimeError('Please set ' + executable_use)
    p = Popen(code_exe, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stat = p.wait()
    out = p.stdout.read()
    err = p.stderr.read()
    print(mode, ': stat= ', stat, ' out= ', out, ' err=', err)
    if mode in ['density']:
        if '!' in err:
            os.chdir(self.start_dir)
            raise RuntimeError(executable_use + ' exited with a code %s' % err)
    elif stat != 0:
        os.chdir(self.start_dir)
        raise RuntimeError(executable_use + ' exited with a code %d' % stat)