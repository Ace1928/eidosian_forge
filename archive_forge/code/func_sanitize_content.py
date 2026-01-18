from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection
def sanitize_content(data):
    out = re.sub('.*Last configuration change.*\n?', '', data)
    return out