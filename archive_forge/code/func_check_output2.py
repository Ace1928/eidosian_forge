from __future__ import (absolute_import, division, print_function)
from contextlib import contextmanager
import os
import re
import subprocess
import time
import yaml
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.utils.display import Display
from ansible.utils.encrypt import random_password
from ansible.plugins.lookup import LookupBase
from ansible import constants as C
from ansible_collections.community.general.plugins.module_utils._filelock import FileLock
def check_output2(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    if 'stderr' in kwargs:
        raise ValueError('stderr argument not allowed, it will be overridden.')
    if 'input' in kwargs:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        b_inputdata = to_bytes(kwargs['input'], errors='surrogate_or_strict')
        del kwargs['input']
        kwargs['stdin'] = subprocess.PIPE
    else:
        b_inputdata = None
    process = subprocess.Popen(*popenargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    try:
        b_out, b_err = process.communicate(b_inputdata)
    except Exception:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if retcode == 0 and (b'encryption failed: Unusable public key' in b_out or b'encryption failed: Unusable public key' in b_err):
        retcode = 78
    if retcode != 0:
        cmd = kwargs.get('args')
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, to_native(b_out + b_err, errors='surrogate_or_strict'))
    return b_out