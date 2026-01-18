from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def complete_missing_attributes(actual, attrs_list, fill_value=None):
    for attribute in attrs_list:
        if not hasattr(actual, attribute):
            setattr(actual, attribute, fill_value)