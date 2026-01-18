from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setDisks(self, name, disks):
    self.__get_conn()
    counter = 0
    bootselect = False
    for disk in disks:
        if 'bootable' in disk:
            if disk['bootable'] is True:
                bootselect = True
    for disk in disks:
        diskname = name + '_Disk' + str(counter) + '_' + disk.get('name', '').replace('/', '_')
        disksize = disk.get('size', 1)
        diskdomain = disk.get('domain', None)
        if diskdomain is None:
            setMsg('`domain` is a required disk key.')
            setFailed()
            return False
        diskinterface = disk.get('interface', 'virtio')
        diskformat = disk.get('format', 'raw')
        diskallocationtype = disk.get('thin', False)
        diskboot = disk.get('bootable', False)
        if bootselect is False and counter == 0:
            diskboot = True
        DISK = self.conn.get_disk(diskname)
        if DISK is None:
            self.conn.createDisk(name, diskname, disksize, diskdomain, diskinterface, diskformat, diskallocationtype, diskboot)
        else:
            self.conn.set_Disk(diskname, disksize, diskinterface, diskboot)
        checkFail()
        counter += 1
    return True