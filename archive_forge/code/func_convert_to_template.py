from __future__ import absolute_import, division, print_function
import re
import time
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def convert_to_template(self, vm, vmid, timeout, force):
    if getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).status.current.get()['status'] == 'running' and force:
        self.stop_instance(vm, vmid, timeout, force)
    getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).template.post()
    return True