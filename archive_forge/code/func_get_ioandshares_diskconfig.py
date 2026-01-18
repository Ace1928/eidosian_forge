from __future__ import absolute_import, division, print_function
import re
from random import randint
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, \
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_ioandshares_diskconfig(self, disk_spec, disk):
    io_disk_spec = vim.StorageResourceManager.IOAllocationInfo()
    if 'iolimit' in disk:
        io_disk_spec.limit = disk['iolimit']['limit']
        if 'shares' in disk['iolimit']:
            shares_spec = vim.SharesInfo()
            shares_spec.level = disk['iolimit']['shares']['level']
            if shares_spec.level == 'custom':
                shares_spec.shares = disk['iolimit']['shares']['level_value']
            io_disk_spec.shares = shares_spec
        disk_spec.device.storageIOAllocation = io_disk_spec
    if 'shares' in disk:
        shares_spec = vim.SharesInfo()
        shares_spec.level = disk['shares']['level']
        if shares_spec.level == 'custom':
            shares_spec.shares = disk['shares']['level_value']
        io_disk_spec.shares = shares_spec
        disk_spec.device.storageIOAllocation = io_disk_spec
    return disk_spec