from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_nve_name(self):
    """Gets Nve interface name"""
    if self.nve_name is None:
        return False
    if self.nve_name in ['Nve1', 'Nve2']:
        return True
    return False