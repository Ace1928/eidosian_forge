from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_NIC(self, vmname, nicname, newname, vlan, interface):
    NIC = self.get_NIC(vmname, nicname)
    VM = self.get_VM(vmname)
    CLUSTER = self.get_cluster_byid(VM.cluster.id)
    DC = self.get_DC_byid(CLUSTER.data_center.id)
    NETWORK = self.get_network(str(DC.name), vlan)
    checkFail()
    if NIC.name != newname:
        NIC.name = newname
        setMsg('Updating iface name to ' + newname)
        setChanged()
    if str(NIC.network.id) != str(NETWORK.id):
        NIC.set_network(NETWORK)
        setMsg('Updating iface network to ' + vlan)
        setChanged()
    if NIC.interface != interface:
        NIC.interface = interface
        setMsg('Updating iface interface to ' + interface)
        setChanged()
    try:
        NIC.update()
        setMsg('iface has successfully been updated.')
    except Exception as e:
        setMsg('Failed to update the iface.')
        setMsg(str(e))
        setFailed()
        return False
    return True