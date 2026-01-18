from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def createNIC(self, vmname, nicname, vlan, interface):
    VM = self.get_VM(vmname)
    CLUSTER = self.get_cluster_byid(VM.cluster.id)
    DC = self.get_DC_byid(CLUSTER.data_center.id)
    newnic = params.NIC(name=nicname, network=DC.networks.get(name=vlan), interface=interface)
    try:
        VM.nics.add(newnic)
        VM.update()
        setMsg('Successfully added iface ' + nicname)
        setChanged()
    except Exception as e:
        setFailed()
        setMsg('Error attaching ' + nicname + ' iface, please recheck and remove any leftover configuration.')
        setMsg(str(e))
        return False
    try:
        currentnic = VM.nics.get(name=nicname)
        attempt = 1
        while currentnic.active is not True:
            currentnic = VM.nics.get(name=nicname)
            if attempt == 100:
                setMsg('Error, iface %s, state %s' % (nicname, str(currentnic.active)))
                raise Exception()
            else:
                attempt += 1
                time.sleep(2)
        setMsg('The iface  ' + nicname + ' is ready.')
    except Exception as e:
        setFailed()
        setMsg('Error getting the state of ' + nicname + '.')
        setMsg(str(e))
        return False
    return True