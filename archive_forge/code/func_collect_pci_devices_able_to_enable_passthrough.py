from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
def collect_pci_devices_able_to_enable_passthrough(self):
    """
        Collect devices able to enable passthrough based on device id.
        """
    self.hosts_passthrough_pci_devices = []
    for esxi_hostname, value in self.hosts_passthrough_pci_device_id.items():
        pci_devices = []
        for device_id in value['pci_device_ids']:
            for device in value['host_obj'].hardware.pciDevice:
                if device.id == device_id.id:
                    pci_devices.append({'device_name': device.deviceName, 'device_id': device.id, 'passthruEnabled': device_id.passthruEnabled})
        self.hosts_passthrough_pci_devices.append({esxi_hostname: {'host_obj': value['host_obj'], 'pci_devices': pci_devices}})