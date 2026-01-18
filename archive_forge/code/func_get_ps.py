from __future__ import (absolute_import, division, print_function)
import glob
import os
import pickle
import platform
import select
import shlex
import subprocess
import traceback
from ansible.module_utils.six import PY2, b
from ansible.module_utils.common.text.converters import to_bytes, to_text
def get_ps(module, pattern):
    """
    Last resort to find a service by trying to match pattern to programs in memory
    """
    found = False
    if platform.system() == 'SunOS':
        flags = '-ef'
    else:
        flags = 'auxww'
    psbin = module.get_bin_path('ps', True)
    rc, psout, pserr = module.run_command([psbin, flags])
    if rc == 0:
        for line in psout.splitlines():
            if pattern in line:
                found = True
                break
    return found