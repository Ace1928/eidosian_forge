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
def make_redirection_tempfiles():
    dummy, out_redir_name = tempfile.mkstemp(prefix='ansible')
    dummy, err_redir_name = tempfile.mkstemp(prefix='ansible')
    return (out_redir_name, err_redir_name)