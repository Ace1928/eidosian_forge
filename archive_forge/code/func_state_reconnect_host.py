from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_reconnect_host(self):
    """Reconnect host to vCenter"""
    changed = True
    result = None
    if self.module.check_mode:
        result = "Host would be reconnected to vCenter '%s'" % self.vcenter
    else:
        self.reconnect_host(self.host)
        result = "Host reconnected to vCenter '%s'" % self.vcenter
    self.module.exit_json(changed=changed, result=str(result))