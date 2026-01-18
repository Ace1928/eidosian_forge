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
def get_redirected_output(path_name):
    output = []
    with open(path_name, 'r') as fd:
        for line in fd:
            new_line = re.sub('\\x1b\\[.+m', '', line)
            output.append(new_line)
    os.remove(path_name)
    return output