from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def get_immutables_intersection(config_proxy, keys):
    immutables_set = set(config_proxy.immutable_attrs)
    keys_set = set(keys)
    return list(immutables_set & keys_set)