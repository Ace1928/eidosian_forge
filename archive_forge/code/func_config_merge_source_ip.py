from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_merge_source_ip(self, nve_name, source_ip):
    """config nve source ip"""
    if self.is_nve_source_ip_change(nve_name, source_ip):
        cfg_xml = CE_NC_MERGE_NVE_SOURCE_IP_PROTOCOL % (nve_name, source_ip)
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'MERGE_SOURCE_IP')
        self.updates_cmd.append('interface %s' % nve_name)
        self.updates_cmd.append('source %s' % source_ip)
        self.changed = True