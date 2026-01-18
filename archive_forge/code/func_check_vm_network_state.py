from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_vm_network_state(self):
    try:
        self.vm = self.find_vm_by_name()
        if self.vm is None:
            self.module.fail_json(msg='A virtual machine with name %s does not exist' % self.vm_name)
        for device in self.vm.config.hardware.device:
            if isinstance(device, vim.vm.device.VirtualEthernetCard):
                if isinstance(device.backing, vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo):
                    return 'present'
        return 'absent'
    except vmodl.RuntimeFault as runtime_fault:
        self.module.fail_json(msg=runtime_fault.msg)
    except vmodl.MethodFault as method_fault:
        self.module.fail_json(msg=method_fault.msg)