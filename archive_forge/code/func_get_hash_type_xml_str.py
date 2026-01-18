from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_hash_type_xml_str(self):
    """trunk hash type netconf xml format string"""
    return HASH_CLI2XML.get(self.hash_type)