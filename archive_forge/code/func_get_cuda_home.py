import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_cuda_home(*subdirs):
    """Get paths of CUDA_HOME.
    If *subdirs* are the subdirectory name to be appended in the resulting
    path.
    """
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is None:
        cuda_home = os.environ.get('CUDA_PATH')
    if cuda_home is not None:
        return os.path.join(cuda_home, *subdirs)