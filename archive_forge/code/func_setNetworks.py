from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setNetworks(self, vmname, ifaces):
    self.__get_conn()
    VM = self.conn.get_VM(vmname)
    counter = 0
    length = len(ifaces)
    for NIC in VM.nics.list():
        if counter < length:
            iface = ifaces[counter]
            name = iface.get('name', None)
            if name is None:
                setMsg('`name` is a required iface key.')
                setFailed()
            elif str(name) != str(NIC.name):
                setMsg('ifaces are in the wrong order, rebuilding everything.')
                for NIC in VM.nics.list():
                    self.conn.del_NIC(vmname, NIC.name)
                self.setNetworks(vmname, ifaces)
                checkFail()
                return True
            vlan = iface.get('vlan', None)
            if vlan is None:
                setMsg('`vlan` is a required iface key.')
                setFailed()
            checkFail()
            interface = iface.get('interface', 'virtio')
            self.conn.set_NIC(vmname, str(NIC.name), name, vlan, interface)
        else:
            self.conn.del_NIC(vmname, NIC.name)
        counter += 1
        checkFail()
    while counter < length:
        iface = ifaces[counter]
        name = iface.get('name', None)
        if name is None:
            setMsg('`name` is a required iface key.')
            setFailed()
        vlan = iface.get('vlan', None)
        if vlan is None:
            setMsg('`vlan` is a required iface key.')
            setFailed()
        if failed is True:
            return False
        interface = iface.get('interface', 'virtio')
        self.conn.createNIC(vmname, name, vlan, interface)
        counter += 1
        checkFail()
    return True