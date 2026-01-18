from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def is_exist_filter(self, filter_feature_name, filter_log_name):
    """if filter info exist"""
    if not self.filter_info:
        return False
    for id2name in self.filter_info['filterInfos']:
        if id2name['icFeatureName'] == filter_feature_name and id2name['icFilterLogName'] == filter_log_name:
            return True
    return False