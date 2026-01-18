from __future__ import absolute_import, division, print_function
import os
import re
import sys
import tempfile
import traceback
from contextlib import contextmanager
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
@contextmanager
def stderr_redirector(path_name):
    old_fh = sys.stderr
    fd = open(path_name, 'w')
    sys.stderr = fd
    try:
        yield
    finally:
        sys.stderr = old_fh