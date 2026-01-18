from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def get_igmp_type_num(self):
    """ Get igmp type num by type """
    if self.igmp_type:
        self.igmp_type_num = IGMP_TYPE_NUM.get(self.igmp_type)