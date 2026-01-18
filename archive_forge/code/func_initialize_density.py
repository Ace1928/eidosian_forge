import os
from subprocess import Popen, PIPE
import re
import numpy as np
from ase.units import Hartree, Bohr
from ase.calculators.calculator import PropertyNotImplementedError
def initialize_density(self, atoms):
    """Creates a new starting density."""
    os.chdir(self.workdir)
    files2remove = ['cdn1', 'fl7para', 'stars', 'wkf2', 'enpara', 'kpts', 'broyd', 'broyd.7', 'tmat', 'tmas']
    if 0:
        files2remove.remove('kpts')
    for f in files2remove:
        if os.path.isfile(f):
            os.remove(f)
    os.system("sed -i -e 's/strho=./strho=T/' inp")
    self.run_executable(mode='density', executable='FLEUR_SERIAL')
    os.system("sed -i -e 's/strho=./strho=F/' inp")
    os.chdir(self.start_dir)
    if atoms.get_initial_magnetic_moments().sum() > 0.0:
        os.chdir(self.workdir)
        os.system("sed -i -e 's/itmax=.*,maxiter/itmax= 1,maxiter/' inp")
        self.run_executable(mode='cdnc', executable='FLEUR')
        sedline = "'s/itmax=.*,maxiter/itmax= '"
        sedline += str(self.itmax_step_default) + "',maxiter/'"
        os.system('sed -i -e ' + sedline + ' inp')
        os.system("sed -i -e 's/swsp=./swsp=T/' inp")
        self.run_executable(mode='swsp', executable='FLEUR_SERIAL')
        os.system("sed -i -e 's/swsp=./swsp=F/' inp")
        os.chdir(self.start_dir)