from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
def check_whether_devices_exist(self):
    """
        Check specified pci devices are exists.
        """
    self.existent_devices = []
    self.non_existent_devices = []
    keys = ['device_name', 'device_id']
    for host_pci_device in self.hosts_passthrough_pci_devices:
        pci_devices = []
        for esxi_hostname, value in host_pci_device.items():
            for target_device in self.devices:
                device = target_device['device']
                if device in [pci_device.get(key) for key in keys for pci_device in value['pci_devices']]:
                    pci_devices.append([pci_device for pci_device in value['pci_devices'] if device == pci_device['device_name'] or device == pci_device['device_id']])
                else:
                    self.non_existent_devices.append(device)
            self.existent_devices.append({esxi_hostname: {'host_obj': value['host_obj'], 'checked_pci_devices': self.de_duplication(sum(pci_devices, []))}})