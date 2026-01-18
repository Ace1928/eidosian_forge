from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback
def is_powered_on(self):
    return self.status == 'active'