from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_valid_spf_max_interval(self):
    """check whether the input ospf spf intelligent timer max interval is valid"""
    if not self.spfmaxinterval.isdigit():
        return False
    if int(self.spfmaxinterval) > 20000 or int(self.spfmaxinterval) < 1:
        return False
    return True