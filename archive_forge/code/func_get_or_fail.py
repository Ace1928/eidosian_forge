from __future__ import (absolute_import, division, print_function)
import uuid
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def get_or_fail(params, key):
    item = params.get(key)
    if item is None:
        raise Exception('{0} must be specified for new volume'.format(key))
    return item