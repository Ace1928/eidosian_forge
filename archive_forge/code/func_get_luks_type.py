from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def get_luks_type(self, device):
    """ get the luks type of a device
        """
    if self.is_luks(device):
        with open(device, 'rb') as f:
            for offset in LUKS2_HEADER_OFFSETS:
                f.seek(offset)
                data = f.read(LUKS_HEADER_L)
                if data == LUKS2_HEADER2:
                    return 'luks2'
            return 'luks1'
    return None