from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def create_eth_trunk(self):
    """Create Eth-Trunk interface"""
    xml_str = CE_NC_XML_CREATE_TRUNK % self.trunk_id
    self.updates_cmd.append('interface Eth-Trunk %s' % self.trunk_id)
    if self.hash_type:
        self.updates_cmd.append('load-balance %s' % self.hash_type)
        xml_str += CE_NC_XML_MERGE_HASHTYPE % (self.trunk_id, self.get_hash_type_xml_str())
    if self.mode:
        self.updates_cmd.append('mode %s' % self.mode)
        xml_str += CE_NC_XML_MERGE_WORKMODE % (self.trunk_id, self.get_mode_xml_str())
    if self.min_links:
        self.updates_cmd.append('least active-linknumber %s' % self.min_links)
        xml_str += CE_NC_XML_MERGE_MINUPNUM % (self.trunk_id, self.min_links)
    if self.members:
        mem_xml = ''
        for mem in self.members:
            mem_xml += CE_NC_XML_MERGE_MEMBER % mem.upper()
            self.updates_cmd.append('interface %s' % mem)
            self.updates_cmd.append('eth-trunk %s' % self.trunk_id)
        xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_xml)
    cfg_xml = CE_NC_XML_BUILD_TRUNK_CFG % xml_str
    self.netconf_set_config(cfg_xml, 'CREATE_TRUNK')
    self.changed = True