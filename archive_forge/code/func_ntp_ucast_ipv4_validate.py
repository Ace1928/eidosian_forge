from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def ntp_ucast_ipv4_validate(self):
    """Check ntp ucast ipv4 address"""
    addr_list = re.findall('(.*)\\.(.*)\\.(.*)\\.(.*)', self.address)
    if not addr_list:
        self.module.fail_json(msg='Error: Match ip-address fail.')
    value = int(addr_list[0][0]) * 16777216 + int(addr_list[0][1]) * 65536 + int(addr_list[0][2]) * 256 + int(addr_list[0][3])
    if value & 4278190080 == 2130706432 or value & 4026531840 == 4026531840 or value & 4026531840 == 3758096384 or (value == 0):
        return False
    return True