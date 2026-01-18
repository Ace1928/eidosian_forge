import functools
import re
import shlex
import sys
from pathlib import Path
from IPython.core.magic import Magics, magics_class, line_magic
def is_conda_environment(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Return True if the current Python executable is in a conda env"""
        if not Path(sys.prefix, 'conda-meta', 'history').exists():
            raise ValueError('The python kernel does not appear to be a conda environment.  Please use ``%pip install`` instead.')
        return func(*args, **kwargs)
    return wrapper