from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_eth_trunk(self):
    """Create or merge Eth-Trunk"""
    change = False
    xml_str = ''
    self.updates_cmd.append('interface Eth-Trunk %s' % self.trunk_id)
    if self.hash_type and self.get_hash_type_xml_str() != self.trunk_info['hashType']:
        self.updates_cmd.append('load-balance %s' % self.hash_type)
        xml_str += CE_NC_XML_MERGE_HASHTYPE % (self.trunk_id, self.get_hash_type_xml_str())
        change = True
    if self.min_links and self.min_links != self.trunk_info['minUpNum']:
        self.updates_cmd.append('least active-linknumber %s' % self.min_links)
        xml_str += CE_NC_XML_MERGE_MINUPNUM % (self.trunk_id, self.min_links)
        change = True
    if self.mode and self.get_mode_xml_str() != self.trunk_info['workMode']:
        self.updates_cmd.append('mode %s' % self.mode)
        xml_str += CE_NC_XML_MERGE_WORKMODE % (self.trunk_id, self.get_mode_xml_str())
        change = True
    if not change:
        self.updates_cmd.pop()
    if self.force and self.trunk_info['TrunkMemberIfs']:
        mem_xml = ''
        for mem in self.trunk_info['TrunkMemberIfs']:
            if not self.members or mem['memberIfName'].replace(' ', '').upper() not in self.members:
                mem_xml += CE_NC_XML_DELETE_MEMBER % mem['memberIfName']
                self.updates_cmd.append('interface %s' % mem['memberIfName'])
                self.updates_cmd.append('undo eth-trunk')
        if mem_xml:
            xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_xml)
            change = True
    if self.members:
        mem_xml = ''
        for mem in self.members:
            if not self.is_member_exist(mem):
                mem_xml += CE_NC_XML_MERGE_MEMBER % mem.upper()
                self.updates_cmd.append('interface %s' % mem)
                self.updates_cmd.append('eth-trunk %s' % self.trunk_id)
        if mem_xml:
            xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_xml)
            change = True
    if not change:
        return
    cfg_xml = CE_NC_XML_BUILD_TRUNK_CFG % xml_str
    self.netconf_set_config(cfg_xml, 'MERGE_TRUNK')
    self.changed = True