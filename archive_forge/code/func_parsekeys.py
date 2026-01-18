from __future__ import absolute_import, division, print_function
import os
import pwd
import os.path
import tempfile
import re
import shlex
from operator import itemgetter
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def parsekeys(module, lines):
    keys = {}
    for rank_index, line in enumerate(lines.splitlines(True)):
        key_data = parsekey(module, line, rank=rank_index)
        if key_data:
            keys[key_data[0]] = key_data
        else:
            keys[line] = (line, 'skipped', None, None, rank_index)
    return keys