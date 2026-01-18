from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def shutdown_instance(self, vm, vmid, timeout):
    taskid = self.vmstatus(vm, vmid).shutdown.post()
    while timeout:
        if self.api_task_ok(vm['node'], taskid):
            return True
        timeout -= 1
        if timeout == 0:
            self.module.fail_json(msg='Reached timeout while waiting for VM to stop. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
        time.sleep(1)
    return False