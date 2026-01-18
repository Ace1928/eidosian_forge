from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_exist_spf_start_interval(self):
    """get exist spf start interval"""
    if not self.ospf_info:
        return None
    for ospf_site in self.ospf_info['ospfsite']:
        if ospf_site['processId'] == self.ospf:
            return ospf_site['spfScheduleStartInterval']
        else:
            continue
    return None