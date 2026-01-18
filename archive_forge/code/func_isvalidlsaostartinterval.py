from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def isvalidlsaostartinterval(self):
    """check whether the input ospf lsa originate start interval is valid"""
    if not self.lsaostartinterval.isdigit():
        return False
    if int(self.lsaostartinterval) > 1000 or int(self.lsaostartinterval) < 0:
        return False
    return True