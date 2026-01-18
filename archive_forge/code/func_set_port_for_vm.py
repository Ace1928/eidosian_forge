from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def set_port_for_vm(self):
    """Sets the ports, to the VM's specified port."""
    if hasattr(self, 'source_vm_transmitted_name') and hasattr(self, 'source_vm_transmitted_nic_label'):
        port = self.get_vm_port(self.source_vm_transmitted_name, self.source_vm_transmitted_nic_label)
        if port is not None:
            self.source_port_transmitted = port
        else:
            self.module.fail_json(msg='No port could be found for VM: {0:s} NIC: {1:s}'.format(self.source_vm_transmitted_name, self.source_vm_transmitted_nic_label))
    if hasattr(self, 'source_vm_received_name') and hasattr(self, 'source_vm_received_nic_label'):
        port = self.get_vm_port(self.source_vm_received_name, self.source_vm_received_nic_label)
        if port is not None:
            self.source_port_received = port
        else:
            self.module.fail_json(msg='No port could be found for VM: {0:s} NIC: {1:s}'.format(self.source_vm_received_name, self.source_vm_received_nic_label))
    if hasattr(self, 'destination_vm_name') and hasattr(self, 'destination_vm_nic_label'):
        port = self.get_vm_port(self.destination_vm_name, self.destination_vm_nic_label)
        if port is not None:
            self.destination_port = port
        else:
            self.module.fail_json(msg='No port could be found for VM: {0:s} NIC: {1:s}'.format(self.destination_vm_name, self.destination_vm_nic_label))