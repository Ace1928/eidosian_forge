from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def get_protocol_num(self):
    """ Get protocol num by name """
    if self.protocol:
        self.protocol_num = PROTOCOL_NUM.get(self.protocol)