from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def createDisk(self, vmname, diskname, disksize, diskdomain, diskinterface, diskformat, diskallocationtype, diskboot):
    VM = self.get_VM(vmname)
    newdisk = params.Disk(name=diskname, size=1024 * 1024 * 1024 * int(disksize), wipe_after_delete=True, sparse=diskallocationtype, interface=diskinterface, format=diskformat, bootable=diskboot, storage_domains=params.StorageDomains(storage_domain=[self.get_domain(diskdomain)]))
    try:
        VM.disks.add(newdisk)
        VM.update()
        setMsg('Successfully added disk ' + diskname)
        setChanged()
    except Exception as e:
        setFailed()
        setMsg('Error attaching ' + diskname + 'disk, please recheck and remove any leftover configuration.')
        setMsg(str(e))
        return False
    try:
        currentdisk = VM.disks.get(name=diskname)
        attempt = 1
        while currentdisk.status.state != 'ok':
            currentdisk = VM.disks.get(name=diskname)
            if attempt == 100:
                setMsg('Error, disk %s, state %s' % (diskname, str(currentdisk.status.state)))
                raise Exception()
            else:
                attempt += 1
                time.sleep(2)
        setMsg('The disk  ' + diskname + ' is ready.')
    except Exception as e:
        setFailed()
        setMsg('Error getting the state of ' + diskname + '.')
        setMsg(str(e))
        return False
    return True