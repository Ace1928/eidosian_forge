from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def is_vrf_exist(self):
    """ judge whether the VPN instance is existed"""
    conf_str = CE_NC_GET_VRF % self.vrf
    con_obj = get_nc_config(self.module, conf_str)
    if '<data/>' in con_obj:
        return False
    return True