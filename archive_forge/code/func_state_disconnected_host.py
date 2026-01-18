from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_disconnected_host(self):
    """Disconnect host to vCenter"""
    changed = True
    result = None
    if self.module.check_mode:
        if self.host.runtime.connectionState == 'disconnected':
            result = 'Host already disconnected'
            changed = False
        else:
            result = "Host would be disconnected host from vCenter '%s'" % self.vcenter
    elif self.host.runtime.connectionState == 'disconnected':
        changed = False
        result = 'Host already disconnected'
    else:
        self.disconnect_host(self.host)
        result = "Host disconnected from vCenter '%s'" % self.vcenter
    self.module.exit_json(changed=changed, result=to_native(result))