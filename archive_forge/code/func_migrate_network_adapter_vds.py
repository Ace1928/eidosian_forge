from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def migrate_network_adapter_vds(self):
    vm_configspec = vim.vm.ConfigSpec()
    nic = vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo()
    port = vim.dvs.PortConnection()
    devicespec = vim.vm.device.VirtualDeviceSpec()
    pg = self.find_dvspg_by_name()
    if pg is None:
        self.module.fail_json(msg='The standard portgroup was not found')
    dvswitch = pg.config.distributedVirtualSwitch
    port.switchUuid = dvswitch.uuid
    port.portgroupKey = pg.key
    nic.port = port
    for device in self.vm.config.hardware.device:
        if isinstance(device, vim.vm.device.VirtualEthernetCard):
            devicespec.device = device
            devicespec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
            devicespec.device.backing = nic
            vm_configspec.deviceChange.append(devicespec)
    task = self.vm.ReconfigVM_Task(vm_configspec)
    changed, result = wait_for_task(task)
    self.module.exit_json(changed=changed, result=result)