from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_mode_xml_str(self):
    """trunk mode netconf xml format string"""
    return MODE_CLI2XML.get(self.mode)