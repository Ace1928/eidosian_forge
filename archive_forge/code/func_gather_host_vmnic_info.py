from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, get_all_objs
def gather_host_vmnic_info(self):
    """Gather vmnic info"""
    hosts_vmnic_info = {}
    for host in self.hosts:
        host_vmnic_info = dict(all=[], available=[], used=[], vswitch=dict(), dvswitch=dict())
        host_nw_system = host.configManager.networkSystem
        if host_nw_system:
            nw_config = host_nw_system.networkConfig
            vmnics = [pnic.device for pnic in nw_config.pnic if pnic.device.startswith('vmnic')]
            host_vmnic_info['all'] = [pnic.device for pnic in nw_config.pnic]
            host_vmnic_info['num_vmnics'] = len(vmnics)
            host_vmnic_info['vmnic_details'] = []
            for pnic in host.config.network.pnic:
                pnic_info = dict()
                if pnic.device.startswith('vmnic'):
                    if pnic.pci:
                        pnic_info['location'] = pnic.pci
                        for pci_device in host.hardware.pciDevice:
                            if pci_device.id == pnic.pci:
                                pnic_info['adapter'] = pci_device.vendorName + ' ' + pci_device.deviceName
                                break
                    else:
                        pnic_info['location'] = 'PCI'
                    pnic_info['device'] = pnic.device
                    pnic_info['driver'] = pnic.driver
                    if pnic.linkSpeed:
                        pnic_info['status'] = 'Connected'
                        pnic_info['actual_speed'] = pnic.linkSpeed.speedMb
                        pnic_info['actual_duplex'] = 'Full Duplex' if pnic.linkSpeed.duplex else 'Half Duplex'
                        try:
                            network_hint = host_nw_system.QueryNetworkHint(pnic.device)
                            for hint in self.to_json(network_hint):
                                if hint.get('lldpInfo'):
                                    pnic_info['lldp_info'] = {x['key']: x['value'] for x in hint['lldpInfo'].get('parameter')}
                                else:
                                    pnic_info['lldp_info'] = 'N/A'
                                if hint.get('connectedSwitchPort'):
                                    pnic_info['cdp_info'] = hint.get('connectedSwitchPort')
                                else:
                                    pnic_info['cdp_info'] = 'N/A'
                        except (vmodl.fault.HostNotConnected, vmodl.fault.HostNotReachable):
                            pnic_info['lldp_info'] = 'N/A'
                            pnic_info['cdp_info'] = 'N/A'
                    else:
                        pnic_info['status'] = 'Disconnected'
                        pnic_info['actual_speed'] = 'N/A'
                        pnic_info['actual_duplex'] = 'N/A'
                        pnic_info['lldp_info'] = 'N/A'
                        pnic_info['cdp_info'] = 'N/A'
                    if pnic.spec.linkSpeed:
                        pnic_info['configured_speed'] = pnic.spec.linkSpeed.speedMb
                        pnic_info['configured_duplex'] = 'Full Duplex' if pnic.spec.linkSpeed.duplex else 'Half Duplex'
                    else:
                        pnic_info['configured_speed'] = 'Auto negotiate'
                        pnic_info['configured_duplex'] = 'Auto negotiate'
                    pnic_info['mac'] = pnic.mac
                    if self.capabilities:
                        pnic_info['nioc_status'] = 'Allowed' if pnic.resourcePoolSchedulerAllowed else 'Not allowed'
                        pnic_info['auto_negotiation_supported'] = pnic.autoNegotiateSupported
                        pnic_info['wake_on_lan_supported'] = pnic.wakeOnLanSupported
                    if self.directpath_io:
                        pnic_info['directpath_io_supported'] = pnic.vmDirectPathGen2Supported
                    if self.directpath_io or self.sriov:
                        if pnic.pci:
                            for pci_device in host.configManager.pciPassthruSystem.pciPassthruInfo:
                                if pci_device.id == pnic.pci:
                                    if self.directpath_io:
                                        pnic_info['passthru_enabled'] = pci_device.passthruEnabled
                                        pnic_info['passthru_capable'] = pci_device.passthruCapable
                                        pnic_info['passthru_active'] = pci_device.passthruActive
                                    if self.sriov:
                                        try:
                                            if pci_device.sriovCapable:
                                                pnic_info['sriov_status'] = 'Enabled' if pci_device.sriovEnabled else 'Disabled'
                                                pnic_info['sriov_active'] = pci_device.sriovActive
                                                pnic_info['sriov_virt_functions'] = pci_device.numVirtualFunction
                                                pnic_info['sriov_virt_functions_requested'] = pci_device.numVirtualFunctionRequested
                                                pnic_info['sriov_virt_functions_supported'] = pci_device.maxVirtualFunctionSupported
                                            else:
                                                pnic_info['sriov_status'] = 'Not supported'
                                        except AttributeError:
                                            pnic_info['sriov_status'] = 'Not supported'
                    host_vmnic_info['vmnic_details'].append(pnic_info)
            vswitch_vmnics = []
            proxy_switch_vmnics = []
            if nw_config.vswitch:
                for vswitch in nw_config.vswitch:
                    host_vmnic_info['vswitch'][vswitch.name] = []
                    try:
                        for vnic in vswitch.spec.bridge.nicDevice:
                            vswitch_vmnics.append(vnic)
                            host_vmnic_info['vswitch'][vswitch.name].append(vnic)
                    except AttributeError:
                        pass
            if nw_config.proxySwitch:
                for proxy_config in nw_config.proxySwitch:
                    dvs_obj = self.find_dvs_by_uuid(uuid=proxy_config.uuid)
                    if dvs_obj:
                        host_vmnic_info['dvswitch'][dvs_obj.name] = []
                    for proxy_nic in proxy_config.spec.backing.pnicSpec:
                        proxy_switch_vmnics.append(proxy_nic.pnicDevice)
                        if dvs_obj:
                            host_vmnic_info['dvswitch'][dvs_obj.name].append(proxy_nic.pnicDevice)
            used_vmics = proxy_switch_vmnics + vswitch_vmnics
            host_vmnic_info['used'] = used_vmics
            host_vmnic_info['available'] = [pnic.device for pnic in nw_config.pnic if pnic.device not in used_vmics]
        hosts_vmnic_info[host.name] = host_vmnic_info
    return hosts_vmnic_info