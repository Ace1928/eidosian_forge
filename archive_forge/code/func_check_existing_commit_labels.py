from __future__ import absolute_import, division, print_function
import json
import re
from difflib import Differ
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def check_existing_commit_labels(conn, label):
    out = conn.get(command='show configuration history detail | include %s' % label)
    label_exist = re.search(label, out, re.M)
    if label_exist:
        return True
    else:
        return False