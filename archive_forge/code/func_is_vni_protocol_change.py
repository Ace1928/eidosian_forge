from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vni_protocol_change(self, nve_name, vni_id, protocol_type):
    """is vni protocol change"""
    if not self.nve_info:
        return True
    if self.nve_info['ifName'] == nve_name:
        for member in self.nve_info['vni_peer_protocols']:
            if member['vniId'] == vni_id and member['protocol'] == protocol_type:
                return False
    return True