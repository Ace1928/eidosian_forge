from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def process_exists_in_guest(self, vm, pid, creds):
    res = self.pm.ListProcessesInGuest(vm, creds, pids=[pid])
    if not res:
        self.module.fail_json(changed=False, msg='ListProcessesInGuest: None (unexpected)')
    res = res[0]
    if res.exitCode is None:
        return (True, None)
    else:
        return (False, res)