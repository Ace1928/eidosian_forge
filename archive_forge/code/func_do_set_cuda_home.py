import sys
import os
import multiprocessing as mp
import warnings
from numba.core.config import IS_WIN32, IS_OSX
from numba.core.errors import NumbaWarning
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import (
from numba.cuda.cuda_paths import (
@staticmethod
def do_set_cuda_home():
    os.environ['CUDA_HOME'] = os.path.join('mycudahome')
    _fake_non_conda_env()
    return (True, _get_cudalib_dir_path_decision())