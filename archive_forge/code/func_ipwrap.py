from __future__ import absolute_import, division, print_function
import types
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
def ipwrap(value, query=''):
    try:
        if isinstance(value, (list, tuple, types.GeneratorType)):
            _ret = []
            for element in value:
                if ipaddr(element, query, version=False, alias='ipwrap'):
                    _ret.append(ipaddr(element, 'wrap'))
                else:
                    _ret.append(element)
            return _ret
        else:
            _ret = ipaddr(value, query, version=False, alias='ipwrap')
            if _ret:
                return ipaddr(_ret, 'wrap')
            else:
                return value
    except Exception:
        return value