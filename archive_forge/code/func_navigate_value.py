from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
def navigate_value(data, index, array_index=None):
    if array_index and (not isinstance(array_index, dict)):
        raise HwcModuleException('array_index must be dict')
    d = data
    for n in range(len(index)):
        if d is None:
            return None
        if not isinstance(d, dict):
            raise HwcModuleException("can't navigate value from a non-dict object")
        i = index[n]
        if i not in d:
            raise HwcModuleException('navigate value failed: key(%s) is not exist in dict' % i)
        d = d[i]
        if not array_index:
            continue
        k = '.'.join(index[:n + 1])
        if k not in array_index:
            continue
        if d is None:
            return None
        if not isinstance(d, list):
            raise HwcModuleException("can't navigate value from a non-list object")
        j = array_index.get(k)
        if j >= len(d):
            raise HwcModuleException('navigate value failed: the index is out of list')
        d = d[j]
    return d