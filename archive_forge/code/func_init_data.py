from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def init_data(self):
    """Init data"""
    if self.interface is not None:
        self.interface = self.interface.lower()
    if not self.key_id:
        self.key_id = ''
    if not self.is_preferred:
        self.is_preferred = 'disable'