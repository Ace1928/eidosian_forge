from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def is_scsi_controller(self, device):
    return isinstance(device, tuple(self.scsi_device_type.values()))