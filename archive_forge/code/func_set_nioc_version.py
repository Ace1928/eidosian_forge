from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def set_nioc_version(self):
    upgrade_spec = vim.DistributedVirtualSwitch.ConfigSpec()
    upgrade_spec.configVersion = self.dvs.config.configVersion
    if not self.version:
        self.version = 'version2'
    upgrade_spec.networkResourceControlVersion = self.version
    try:
        task = self.dvs.ReconfigureDvs_Task(spec=upgrade_spec)
        wait_for_task(task)
    except vmodl.RuntimeFault as runtime_fault:
        self.module.fail_json(msg='RuntimeFault when setting NIOC version: %s ' % to_native(runtime_fault.msg))