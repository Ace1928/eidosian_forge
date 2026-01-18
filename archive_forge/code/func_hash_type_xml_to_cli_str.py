from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def hash_type_xml_to_cli_str(hash_type):
    """convert trunk hash type netconf xml to cli format string"""
    if not hash_type:
        return ''
    return HASH_XML2CLI.get(hash_type)