from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from copy import deepcopy
def get_iscsi_config(self):
    self.existing_system_iscsi_config = {}
    for hba in self.host_obj.config.storageDevice.hostBusAdapter:
        if isinstance(hba, vim.host.InternetScsiHba):
            self.existing_system_iscsi_config.update({'vmhba_name': hba.device, 'iscsi_name': hba.iScsiName, 'iscsi_alias': hba.iScsiAlias, 'iscsi_authentication_properties': self.to_json(hba.authenticationProperties)})
            iscsi_send_targets = []
            for iscsi_send_target in self.to_json(hba.configuredSendTarget):
                iscsi_send_targets.append({'address': iscsi_send_target['address'], 'authenticationProperties': iscsi_send_target['authenticationProperties'], 'port': iscsi_send_target['port']})
            self.existing_system_iscsi_config['iscsi_send_targets'] = iscsi_send_targets
            iscsi_static_targets = []
            for iscsi_static_target in self.to_json(hba.configuredStaticTarget):
                iscsi_static_targets.append({'iscsi_name': iscsi_static_target['iScsiName'], 'address': iscsi_static_target['address'], 'authenticationProperties': iscsi_static_target['authenticationProperties'], 'port': iscsi_static_target['port']})
            self.existing_system_iscsi_config['iscsi_static_targets'] = iscsi_static_targets
    self.existing_system_iscsi_config['iscsi_enabled'] = self.to_json(self.host_obj.config.storageDevice.softwareInternetScsiEnabled)
    vnic_devices = []
    if self.iscsi_config:
        for vnic in self.host_obj.configManager.iscsiManager.QueryBoundVnics(iScsiHbaName=self.vmhba_name):
            vnic_devices.append(vnic.vnicDevice)
    self.existing_system_iscsi_config['port_bind'] = vnic_devices