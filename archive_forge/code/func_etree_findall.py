from __future__ import absolute_import, division, print_function
import json
import re
from difflib import Differ
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def etree_findall(root, node):
    try:
        root = etree.fromstring(to_bytes(root))
    except (ValueError, etree.XMLSyntaxError):
        pass
    return root.findall('.//%s' % node.strip())