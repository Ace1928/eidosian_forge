from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_area_ip(self):
    """convert integer to ip address"""
    if not self.area.isdigit():
        return self.area
    addr_int = ['0'] * 4
    addr_int[0] = str((int(self.area) & 4278190080) >> 24 & 255)
    addr_int[1] = str((int(self.area) & 16711680) >> 16 & 255)
    addr_int[2] = str((int(self.area) & 65280) >> 8 & 255)
    addr_int[3] = str(int(self.area) & 255)
    return '.'.join(addr_int)