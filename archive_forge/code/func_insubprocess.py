import sys
import os
import shutil
import inspect
import tempfile
import subprocess
from contextlib import contextmanager
from functools import wraps
import numpy as np
from numpy.lib.recfunctions import repack_fields
import h5py
import unittest as ut
def insubprocess(f):
    """Runs a test in its own subprocess"""

    @wraps(f)
    def wrapper(request, *args, **kwargs):
        curr_test = inspect.getsourcefile(f) + '::' + request.node.name
        insub = 'IN_SUBPROCESS_' + curr_test
        for c in '/\\,:.':
            insub = insub.replace(c, '_')
        defined = os.environ.get(insub, None)
        if defined:
            return f(request, *args, **kwargs)
        else:
            os.environ[insub] = '1'
            env = os.environ.copy()
            env[insub] = '1'
            env.update(getattr(f, 'subproc_env', {}))
            with closed_tempfile() as stdout:
                with open(stdout, 'w+t') as fh:
                    rtn = subprocess.call([sys.executable, '-m', 'pytest', curr_test], stdout=fh, stderr=fh, env=env)
                with open(stdout, 'rt') as fh:
                    out = fh.read()
            assert rtn == 0, '\n' + out
    return wrapper