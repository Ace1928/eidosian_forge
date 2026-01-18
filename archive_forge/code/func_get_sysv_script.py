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
def get_sysv_script(name):
    """
    This function will return the expected path for an init script
    corresponding to the service name supplied.

    :arg name: name or path of the service to test for
    """
    if name.startswith('/'):
        result = name
    else:
        result = '/etc/init.d/%s' % name
    return result