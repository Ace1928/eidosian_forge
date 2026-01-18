from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def nxosCmdRef_import_check():
    """Return import error messages or empty string"""
    msg = ''
    if not HAS_YAML:
        msg += "Mandatory python library 'PyYAML' is not present, try 'pip install PyYAML'\n"
    return msg