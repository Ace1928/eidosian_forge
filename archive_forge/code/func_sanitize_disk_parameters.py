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
def sanitize_disk_parameters(self, vm_obj):
    """

        Sanitize user provided disk parameters to configure multiple types of disk controllers and attached disks

        Returns: A sanitized dict of disk params, else fails
                 e.g., [{'type': 'nvme', 'num': 1, 'disk': []}, {}, {}, {}]}

        """
    controllers = []
    for disk_spec in self.params.get('disk'):
        if disk_spec['controller_type'] is None or disk_spec['controller_number'] is None or disk_spec['unit_number'] is None:
            self.module.fail_json(msg="'disk.controller_type', 'disk.controller_number' and 'disk.unit_number' are mandatory parameters when configure multiple disk controllers and disks.")
        ctl_num = disk_spec['controller_number']
        ctl_unit_num = disk_spec['unit_number']
        disk_spec['unit_number'] = ctl_unit_num
        ctl_type = disk_spec['controller_type']
        if len(controllers) != 0:
            ctl_exist = False
            for ctl in controllers:
                if ctl['type'] in self.device_helper.scsi_device_type.keys() and ctl_type in self.device_helper.scsi_device_type.keys():
                    if ctl['type'] != ctl_type and ctl['num'] == ctl_num:
                        self.module.fail_json(msg="Specified SCSI controller '%s' and '%s' have the same bus number: '%s'" % (ctl['type'], ctl_type, ctl_num))
                if ctl['type'] == ctl_type and ctl['num'] == ctl_num:
                    for i in range(0, len(ctl['disk'])):
                        if disk_spec['unit_number'] == ctl['disk'][i]['unit_number']:
                            self.module.fail_json(msg="Specified the same 'controller_type, controller_number, unit_number in disk configuration '%s:%s'" % (ctl_type, ctl_num))
                    ctl['disk'].append(disk_spec)
                    ctl_exist = True
                    break
            if not ctl_exist:
                controllers.append({'type': ctl_type, 'num': ctl_num, 'disk': [disk_spec]})
        else:
            controllers.append({'type': ctl_type, 'num': ctl_num, 'disk': [disk_spec]})
    return controllers