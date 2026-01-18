from __future__ import absolute_import, division, print_function
import base64
import time
import os
import traceback
import sys
import hashlib
from datetime import datetime
from tempfile import NamedTemporaryFile
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
from ansible_collections.kubernetes.core.plugins.module_utils.selector import (
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils._text import to_native, to_bytes, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.urls import Request
def load_resource_definitions(self, src, module=None):
    """Load the requested src path"""
    if module and (src.startswith('https://') or src.startswith('http://') or src.startswith('ftp://')):
        src = fetch_file_from_url(module, src)
    result = None
    path = os.path.normpath(src)
    if not os.path.exists(path):
        self.fail(msg='Error accessing {0}. Does the file exist?'.format(path))
    try:
        with open(path, 'rb') as f:
            result = list(yaml.safe_load_all(f))
    except (IOError, yaml.YAMLError) as exc:
        self.fail(msg='Error loading resource_definition: {0}'.format(exc))
    return result