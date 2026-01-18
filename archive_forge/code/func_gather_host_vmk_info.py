from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def gather_host_vmk_info(self):
    hosts_info = {}
    for host in self.hosts:
        host_vmk_info = []
        host_network_system = host.config.network
        if host_network_system:
            vmks_config = host.config.network.vnic
            for vmk in vmks_config:
                host_vmk_info.append(dict(device=vmk.device, key=vmk.key, portgroup=vmk.portgroup, ipv4_address=vmk.spec.ip.ipAddress, ipv4_subnet_mask=vmk.spec.ip.subnetMask, dhcp=vmk.spec.ip.dhcp, mac=vmk.spec.mac, mtu=vmk.spec.mtu, stack=vmk.spec.netStackInstanceKey, enable_vsan=vmk.device in self.service_type_vmks[host.name]['vsan'], enable_vmotion=vmk.device in self.service_type_vmks[host.name]['vmotion'], enable_management=vmk.device in self.service_type_vmks[host.name]['management'], enable_ft=vmk.device in self.service_type_vmks[host.name]['faultToleranceLogging']))
        hosts_info[host.name] = host_vmk_info
    return hosts_info