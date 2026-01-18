from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def isvalidlsastartarrivalinterval(self):
    """check whether the input ospf lsa start arrival interval is valid"""
    if not self.lsaastartinterval.isdigit():
        return False
    if int(self.lsaastartinterval) > 1000 or int(self.lsaastartinterval) < 0:
        return False
    return True