from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_key_or_fail(self, k):
    v = self.module.params[k]
    if v is None:
        self.module.fail_json(msg='Unable to load %s' % k)
    return v