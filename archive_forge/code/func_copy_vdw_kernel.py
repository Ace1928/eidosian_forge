import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def copy_vdw_kernel(self, directory='./'):
    """Method to copy the vdw_kernel.bindat file.
        Set ASE_VASP_VDW environment variable to the vdw_kernel.bindat
        folder location. Checks if LUSE_VDW is enabled, and if no location
        for the vdW kernel is specified, a warning is issued."""
    vdw_env = 'ASE_VASP_VDW'
    kernel = 'vdw_kernel.bindat'
    dst = os.path.join(directory, kernel)
    if isfile(dst):
        return
    if self.bool_params['luse_vdw']:
        src = None
        if vdw_env in os.environ:
            src = os.path.join(os.environ[vdw_env], kernel)
        if not src or not isfile(src):
            warnings.warn('vdW has been enabled, however no location for the {} file has been specified. Set {} environment variable to copy the vdW kernel.'.format(kernel, vdw_env))
        else:
            shutil.copyfile(src, dst)