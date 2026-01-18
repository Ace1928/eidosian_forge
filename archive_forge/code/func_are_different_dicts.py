from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
def are_different_dicts(dict1, dict2):
    return _DictComparison(dict1) != _DictComparison(dict2)