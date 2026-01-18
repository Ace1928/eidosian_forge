from subprocess import check_output
import os.path
from collections import defaultdict
import inspect
from functools import partial
import numba
from numba.core.registry import cpu_target
from all overloads.
def git_hash():
    out = check_output(['git', 'log', "--pretty=format:'%H'", '-n', '1'])
    return out.decode('ascii').strip('\'"')