from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def modify_dvs_host(self, operation):
    changed, result = (False, None)
    spec = vim.DistributedVirtualSwitch.ConfigSpec()
    spec.configVersion = self.dv_switch.config.configVersion
    spec.host = [vim.dvs.HostMember.ConfigSpec()]
    spec.host[0].operation = operation
    spec.host[0].host = self.host
    if self.vendor_specific_config:
        config = list()
        for item in self.vendor_specific_config:
            config.append(vim.dvs.KeyedOpaqueBlob(key=item['key'], opaqueData=item['value']))
        spec.host[0].vendorSpecificConfig = config
    if operation == 'edit':
        spec.host[0].backing = vim.dvs.HostMember.PnicBacking()
        for nic, uplinkPortKey in self.desired_state.items():
            pnicSpec = vim.dvs.HostMember.PnicSpec()
            pnicSpec.pnicDevice = nic
            pnicSpec.uplinkPortgroupKey = self.uplink_portgroup.key
            pnicSpec.uplinkPortKey = uplinkPortKey
            spec.host[0].backing.pnicSpec.append(pnicSpec)
    try:
        task = self.dv_switch.ReconfigureDvs_Task(spec)
        changed, result = wait_for_task(task)
    except vmodl.fault.NotSupported as not_supported:
        self.module.fail_json(msg='Failed to configure DVS host %s as it is not compatible with the VDS version.' % self.esxi_hostname, details=to_native(not_supported.msg))
    return (changed, result)