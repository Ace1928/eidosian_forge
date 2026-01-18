from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def set_disk_parameters(self, disk_spec, expected_disk_spec, reconfigure=False):
    disk_modified = False
    if expected_disk_spec['disk_mode']:
        disk_mode = expected_disk_spec.get('disk_mode')
        if reconfigure:
            if disk_spec.device.backing.diskMode != disk_mode:
                disk_spec.device.backing.diskMode = disk_mode
                disk_modified = True
        else:
            disk_spec.device.backing.diskMode = disk_mode
    elif not reconfigure:
        disk_spec.device.backing.diskMode = 'persistent'
    if not reconfigure:
        disk_type = expected_disk_spec.get('type', 'thin')
        if disk_type == 'thin':
            disk_spec.device.backing.thinProvisioned = True
        elif disk_type == 'eagerzeroedthick':
            disk_spec.device.backing.eagerlyScrub = True
    kb = self.get_configured_disk_size(expected_disk_spec)
    if reconfigure:
        if disk_spec.device.capacityInKB > kb:
            self.module.fail_json(msg='Given disk size is smaller than found (%d < %d).Reducing disks is not allowed.' % (kb, disk_spec.device.capacityInKB))
        if disk_spec.device.capacityInKB != kb:
            disk_spec.device.capacityInKB = kb
            disk_modified = True
    else:
        disk_spec.device.capacityInKB = kb
        disk_modified = True
    return disk_modified