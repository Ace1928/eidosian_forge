from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vni_bd_change(self, vni_id, bd_id):
    """is vni to bridge-domain-id change"""
    if not self.vni2bd_info:
        return True
    for vni2bd in self.vni2bd_info['vni2BdInfos']:
        if vni2bd['vniId'] == vni_id and vni2bd['bdId'] == bd_id:
            return False
    return True