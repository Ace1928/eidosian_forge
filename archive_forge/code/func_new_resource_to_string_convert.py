from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def new_resource_to_string_convert(self, resrc):
    dict_valid_values = dict(((k.replace('_', '', 1), v) for k, v in resrc.__dict__.items() if v))
    return json.dumps(dict_valid_values)