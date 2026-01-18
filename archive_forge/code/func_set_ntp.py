from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def set_ntp(self, *args):
    """Configure ntp parameters"""
    if self.state == 'present':
        if self.ip_ver == 'IPv4':
            xml_str = CE_NC_MERGE_NTP_CONFIG % (args[0], args[1], '::', args[2], args[3], args[4], args[5], args[6])
        elif self.ip_ver == 'IPv6':
            xml_str = CE_NC_MERGE_NTP_CONFIG % (args[0], '0.0.0.0', args[1], args[2], args[3], args[4], args[5], args[6])
        ret_xml = set_nc_config(self.module, xml_str)
        self.check_response(ret_xml, 'NTP_CORE_CONFIG')
    else:
        if self.ip_ver == 'IPv4':
            xml_str = CE_NC_DELETE_NTP_CONFIG % (args[0], args[1], '::', args[2], args[3])
        elif self.ip_ver == 'IPv6':
            xml_str = CE_NC_DELETE_NTP_CONFIG % (args[0], '0.0.0.0', args[1], args[2], args[3])
        ret_xml = set_nc_config(self.module, xml_str)
        self.check_response(ret_xml, 'UNDO_NTP_CORE_CONFIG')