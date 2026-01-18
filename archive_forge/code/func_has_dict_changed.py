from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def has_dict_changed(new_dict, old_dict):
    """
    Check if new_dict has differences compared to old_dict while
    ignoring keys in old_dict which are None in new_dict.
    """
    if new_dict is None:
        return False
    if not new_dict and old_dict:
        return True
    if not old_dict and new_dict:
        return True
    defined_options = dict(((option, value) for option, value in new_dict.items() if value is not None))
    for option, value in defined_options.items():
        old_value = old_dict.get(option)
        if not value and (not old_value):
            continue
        if value != old_value:
            return True
    return False