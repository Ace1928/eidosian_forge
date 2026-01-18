from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_Disk(self, diskname, disksize, diskinterface, diskboot):
    DISK = self.get_disk(diskname)
    setMsg('Checking disk ' + diskname)
    if DISK.get_bootable() != diskboot:
        try:
            DISK.set_bootable(diskboot)
            setMsg('Updated the boot option on the disk.')
            setChanged()
        except Exception as e:
            setMsg('Failed to set the boot option on the disk.')
            setMsg(str(e))
            setFailed()
            return False
    else:
        setMsg('The boot option of the disk is correct')
    if int(DISK.size) < 1024 * 1024 * 1024 * int(disksize):
        try:
            DISK.size = 1024 * 1024 * 1024 * int(disksize)
            setMsg('Updated the size of the disk.')
            setChanged()
        except Exception as e:
            setMsg('Failed to update the size of the disk.')
            setMsg(str(e))
            setFailed()
            return False
    elif int(DISK.size) > 1024 * 1024 * 1024 * int(disksize):
        setMsg('Shrinking disks is not supported')
        setFailed()
        return False
    else:
        setMsg('The size of the disk is correct')
    if str(DISK.interface) != str(diskinterface):
        try:
            DISK.interface = diskinterface
            setMsg('Updated the interface of the disk.')
            setChanged()
        except Exception as e:
            setMsg('Failed to update the interface of the disk.')
            setMsg(str(e))
            setFailed()
            return False
    else:
        setMsg('The interface of the disk is correct')
    return True