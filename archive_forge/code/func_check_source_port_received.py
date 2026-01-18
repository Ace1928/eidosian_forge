from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_source_port_received(self, session):
    if not self.source_port_received:
        return
    if self.session_type == 'remoteMirrorDest':
        port = vim.dvs.VmwareDistributedVirtualSwitch.VspanPorts(vlans=[int(self.source_port_received)])
        if int(self.source_port_received) not in self.dv_switch.QueryUsedVlanIdInDvs():
            self.module.fail_json(msg="Couldn't find vlan: {0:s}".format(self.source_port_received))
        session.sourcePortReceived = port
    else:
        port = vim.dvs.VmwareDistributedVirtualSwitch.VspanPorts(portKey=str(self.source_port_received))
        if not self.dv_switch.FetchDVPorts(vim.dvs.PortCriteria(portKey=port.portKey)):
            self.module.fail_json(msg="Couldn't find port: {0:s}".format(self.source_port_received))
        session.sourcePortReceived = port