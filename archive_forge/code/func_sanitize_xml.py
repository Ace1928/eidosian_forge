from __future__ import absolute_import, division, print_function
import json
from contextlib import contextmanager
from copy import deepcopy
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
def sanitize_xml(data):
    tree = fromstring(to_bytes(deepcopy(data), errors='surrogate_then_replace'))
    for element in tree.iter():
        attribute = element.attrib
        if attribute:
            for key in list(attribute):
                if key not in IGNORE_XML_ATTRIBUTE:
                    attribute.pop(key)
    return to_text(tostring(tree), errors='surrogate_then_replace').strip()