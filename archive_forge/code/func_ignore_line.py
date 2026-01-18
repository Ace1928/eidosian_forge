from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_native, to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, ConfigLine
def ignore_line(text, tokens=None):
    for item in tokens or DEFAULT_COMMENT_TOKENS:
        if text.startswith(item):
            return True
    for regex in DEFAULT_IGNORE_LINES_RE:
        if regex.match(text):
            return True