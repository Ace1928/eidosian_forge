from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_nve_source_ip_change(self, nve_name, source_ip):
    """is vni to bridge-domain-id change"""
    if not self.nve_info:
        return True
    if self.nve_info['ifName'] == nve_name and self.nve_info['srcAddr'] == source_ip:
        return False
    return True