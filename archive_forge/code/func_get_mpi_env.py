from __future__ import absolute_import
import sys
import subprocess, logging
from threading import Thread
from . import tracker
def get_mpi_env(envs):
    """get the mpirun command for setting the envornment
    support both openmpi and mpich2
    """
    cmd = ''
    if sys.platform == 'win32':
        for k, v in envs.items():
            cmd += ' -env %s %s' % (k, str(v))
        return cmd
    out, err = subprocess.Popen(['mpirun', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    if b'Open MPI' in out:
        for k, v in envs.items():
            cmd += ' -x %s=%s' % (k, str(v))
    elif b'mpich' in out:
        for k, v in envs.items():
            cmd += ' -env %s %s' % (k, str(v))
    else:
        raise RuntimeError('Unknown MPI Version')
    return cmd