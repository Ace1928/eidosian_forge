from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def host_vmk_create(self):
    """
        Create VMKernel
        Returns: NA

        """
    results = dict(changed=False, message='')
    if self.vswitch_name:
        results['switch'] = self.vswitch_name
    elif self.vds_name:
        results['switch'] = self.vds_name
    results['portgroup'] = self.port_group_name
    vnic_config = vim.host.VirtualNic.Specification()
    ip_spec = vim.host.IpConfig()
    results['ipv4'] = self.network_type
    if self.network_type == 'dhcp':
        ip_spec.dhcp = True
    else:
        ip_spec.dhcp = False
        results['ipv4_ip'] = self.ip_address
        results['ipv4_sm'] = self.subnet_mask
        ip_spec.ipAddress = self.ip_address
        ip_spec.subnetMask = self.subnet_mask
        if self.default_gateway:
            vnic_config.ipRouteSpec = vim.host.VirtualNic.IpRouteSpec()
            vnic_config.ipRouteSpec.ipRouteConfig = vim.host.IpRouteConfig()
            vnic_config.ipRouteSpec.ipRouteConfig.defaultGateway = self.default_gateway
    vnic_config.ip = ip_spec
    results['mtu'] = self.mtu
    vnic_config.mtu = self.mtu
    results['tcpip_stack'] = self.tcpip_stack
    vnic_config.netStackInstanceKey = self.get_api_net_stack_instance(self.tcpip_stack)
    vmk_device = None
    try:
        if self.module.check_mode:
            results['msg'] = 'VMkernel Adapter would be created'
        else:
            if self.vswitch_name:
                vmk_device = self.esxi_host_obj.configManager.networkSystem.AddVirtualNic(self.port_group_name, vnic_config)
            elif self.vds_name:
                vnic_config.distributedVirtualPort = vim.dvs.PortConnection()
                vnic_config.distributedVirtualPort.switchUuid = self.dv_switch_obj.uuid
                vnic_config.distributedVirtualPort.portgroupKey = self.port_group_obj.key
                vmk_device = self.esxi_host_obj.configManager.networkSystem.AddVirtualNic(portgroup='', nic=vnic_config)
            results['msg'] = 'VMkernel Adapter created'
        results['changed'] = True
        results['device'] = vmk_device
        if self.network_type != 'dhcp':
            if self.default_gateway:
                results['ipv4_gw'] = self.default_gateway
            else:
                results['ipv4_gw'] = 'No override'
        results['services'] = self.create_enabled_services_string()
    except vim.fault.AlreadyExists as already_exists:
        self.module.fail_json(msg='Failed to add vmk as portgroup already has a virtual network adapter %s' % to_native(already_exists.msg))
    except vim.fault.HostConfigFault as host_config_fault:
        self.module.fail_json(msg='Failed to add vmk due to host config issues : %s' % to_native(host_config_fault.msg))
    except vim.fault.InvalidState as invalid_state:
        self.module.fail_json(msg='Failed to add vmk as ipv6 address is specified in an ipv4 only system : %s' % to_native(invalid_state.msg))
    except vmodl.fault.InvalidArgument as invalid_arg:
        self.module.fail_json(msg='Failed to add vmk as IP address or Subnet Mask in the IP configuration are invalid or PortGroup does not exist : %s' % to_native(invalid_arg.msg))
    if self.tcpip_stack == 'default' and (not all((option is False for option in [self.enable_vsan, self.enable_vmotion, self.enable_mgmt, self.enable_ft, self.enable_provisioning, self.enable_replication, self.enable_replication_nfc, self.enable_backup_nfc]))):
        self.vnic = self.get_vmkernel_by_device(device_name=vmk_device)
        if self.enable_vsan:
            results['vsan'] = self.set_vsan_service_type(self.enable_vsan)
        host_vnic_manager = self.esxi_host_obj.configManager.virtualNicManager
        if self.enable_vmotion:
            self.set_service_type(host_vnic_manager, self.vnic, 'vmotion')
        if self.enable_mgmt:
            self.set_service_type(host_vnic_manager, self.vnic, 'management')
        if self.enable_ft:
            self.set_service_type(host_vnic_manager, self.vnic, 'faultToleranceLogging')
        if self.enable_provisioning:
            self.set_service_type(host_vnic_manager, self.vnic, 'vSphereProvisioning')
        if self.enable_replication:
            self.set_service_type(host_vnic_manager, self.vnic, 'vSphereReplication')
        if self.enable_replication_nfc:
            self.set_service_type(host_vnic_manager, self.vnic, 'vSphereReplicationNFC')
        if self.enable_backup_nfc:
            self.set_service_type(host_vnic_manager, self.vnic, 'vSphereBackupNFC')
    self.module.exit_json(**results)