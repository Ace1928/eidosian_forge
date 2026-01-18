from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def isvalidlsamaxarrivalinterval(self):
    """check whether the input ospf lsa max arrival interval is valid"""
    if not self.lsaamaxinterval.isdigit():
        return False
    if int(self.lsaamaxinterval) > 10000 or int(self.lsaamaxinterval) < 1:
        return False
    return True