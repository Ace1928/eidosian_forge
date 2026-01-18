from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def parse_openssh_version(version_string):
    """Parse the version output of ssh -V and return version numbers that can be compared"""
    parsed_result = re.match('^.*openssh_(?P<version>[0-9.]+)(p?[0-9]+)[^0-9]*.*$', version_string.lower())
    if parsed_result is not None:
        version = parsed_result.group('version').strip()
    else:
        version = None
    return version