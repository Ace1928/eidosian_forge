from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_valid_spf_hold_interval(self):
    """check whether the input ospf spf intelligent timer hold interval is valid"""
    if not self.spfholdinterval.isdigit():
        return False
    if int(self.spfholdinterval) > 5000 or int(self.spfholdinterval) < 1:
        return False
    return True