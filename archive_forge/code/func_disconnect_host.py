from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def disconnect_host(self, host_object):
    """Disconnect host to vCenter"""
    try:
        task = host_object.DisconnectHost_Task()
    except Exception as e:
        self.module.fail_json(msg='Failed to disconnect host from vCenter: %s' % to_native(e))
    try:
        changed, result = wait_for_task(task)
    except TaskError as task_error:
        self.module.fail_json(msg="Failed to disconnect host from vCenter '%s' due to %s" % (self.vcenter, to_native(task_error)))