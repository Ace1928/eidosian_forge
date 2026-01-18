from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_source_port_transmitted(self, session):
    if self.source_port_transmitted is not None:
        port = vim.dvs.VmwareDistributedVirtualSwitch.VspanPorts(portKey=str(self.source_port_transmitted))
        if not self.dv_switch.FetchDVPorts(vim.dvs.PortCriteria(portKey=port.portKey)):
            self.module.fail_json(msg="Couldn't find port: {0:s}".format(self.source_port_transmitted))
        session.sourcePortTransmitted = port