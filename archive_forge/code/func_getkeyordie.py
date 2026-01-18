from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback
def getkeyordie(k):
    v = module.params[k]
    if v is None:
        module.fail_json(msg='Unable to load %s' % k)
    return v