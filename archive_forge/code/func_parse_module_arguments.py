from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
def parse_module_arguments(module_arguments, return_none=False):
    if module_arguments is None:
        return None if return_none else []
    if isinstance(module_arguments, list) and len(module_arguments) == 1 and (not module_arguments[0]):
        return []
    if not isinstance(module_arguments, list):
        module_arguments = [module_arguments]
    parsed_args = []
    re_clear_spaces = re.compile('\\s*=\\s*')
    for arg in module_arguments:
        for item in filter(None, RULE_ARG_REGEX.findall(arg)):
            if not item.startswith('['):
                re_clear_spaces.sub('=', item)
            parsed_args.append(item)
    return parsed_args