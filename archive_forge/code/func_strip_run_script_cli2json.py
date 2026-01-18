from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible.plugins.httpapi import HttpApiBase
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
from ansible.module_utils.common._collections_compat import Mapping
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
def strip_run_script_cli2json(command):
    if to_text(command, errors='surrogate_then_replace').startswith('run script cli2json.py'):
        command = str(command).replace('run script cli2json.py', '')
    return command