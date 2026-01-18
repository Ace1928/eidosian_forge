from __future__ import absolute_import, division, print_function
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def is_valid_ip_vpn(vpname):
    """check ip vpn"""
    if not vpname:
        return False
    if vpname == '_public_':
        return False
    if len(vpname) < 1 or len(vpname) > 31:
        return False
    return True