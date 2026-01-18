import sys
import os
import os.path as op
import tempfile
from subprocess import Popen, check_output, PIPE, STDOUT, CalledProcessError
from srsly.cloudpickle.compat import pickle
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import psutil
from srsly.cloudpickle import dumps
from subprocess import TimeoutExpired
def subprocess_pickle_echo(input_data, protocol=None, timeout=TIMEOUT, add_env=None):
    """Echo function with a child Python process
    Pickle the input data into a buffer, send it to a subprocess via
    stdin, expect the subprocess to unpickle, re-pickle that data back
    and send it back to the parent process via stdout for final unpickling.
    >>> subprocess_pickle_echo([1, 'a', None])
    [1, 'a', None]
    """
    out = subprocess_pickle_string(input_data, protocol=protocol, timeout=timeout, add_env=add_env)
    return loads(out)