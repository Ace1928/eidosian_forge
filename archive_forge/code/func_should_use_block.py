from __future__ import (absolute_import, division, print_function)
import yaml
import json
import re
import string
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.plugins.callback import strip_internal_keys, module_response_deepcopy
from ansible.plugins.callback.default import CallbackModule as Default
def should_use_block(value):
    """Returns true if string should be in block format"""
    for c in u'\n\r\x1c\x1d\x1e\x85\u2028\u2029':
        if c in value:
            return True
    return False