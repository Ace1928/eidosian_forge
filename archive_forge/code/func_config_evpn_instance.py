from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_evpn_instance(self):
    """Configure EVPN instance"""
    self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
    if self.evpn == 'disable':
        xml_str = CE_NC_DELETE_EVPN_CONFIG % (self.bridge_domain_id, self.bridge_domain_id)
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'MERGE_EVPN_BD_CONFIG')
        self.updates_cmd.append('  undo evpn')
        self.changed = True
        return
    xml_str = CE_NC_MERGE_EVPN_CONFIG_HEAD % (self.bridge_domain_id, self.bridge_domain_id)
    self.updates_cmd.append('  evpn')
    if self.route_distinguisher:
        if not self.existing['route_distinguisher']:
            if self.route_distinguisher.lower() == 'auto':
                xml_str += '<evpnAutoRD>true</evpnAutoRD>'
                self.updates_cmd.append('    route-distinguisher auto')
            else:
                xml_str += '<evpnRD>%s</evpnRD>' % self.route_distinguisher
                self.updates_cmd.append('    route-distinguisher %s' % self.route_distinguisher)
    vpn_target_export = copy.deepcopy(self.vpn_target_export)
    vpn_target_import = copy.deepcopy(self.vpn_target_import)
    if self.vpn_target_both:
        for ele in self.vpn_target_both:
            if ele not in vpn_target_export:
                vpn_target_export.append(ele)
            if ele not in vpn_target_import:
                vpn_target_import.append(ele)
    head_flag = False
    if vpn_target_export:
        for ele in vpn_target_export:
            if ele.lower() == 'auto' and (not self.is_vpn_target_exist('export_extcommunity', ele.lower())):
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_AUTORTS_HEAD
                    head_flag = True
                xml_str += CE_NC_MERGE_EVPN_AUTORTS_CONTEXT % 'export_extcommunity'
                self.updates_cmd.append('    vpn-target auto export-extcommunity')
    if vpn_target_import:
        for ele in vpn_target_import:
            if ele.lower() == 'auto' and (not self.is_vpn_target_exist('import_extcommunity', ele.lower())):
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_AUTORTS_HEAD
                    head_flag = True
                xml_str += CE_NC_MERGE_EVPN_AUTORTS_CONTEXT % 'import_extcommunity'
                self.updates_cmd.append('    vpn-target auto import-extcommunity')
    if head_flag:
        xml_str += CE_NC_MERGE_EVPN_AUTORTS_TAIL
    head_flag = False
    if vpn_target_export:
        for ele in vpn_target_export:
            if ele.lower() != 'auto' and (not self.is_vpn_target_exist('export_extcommunity', ele.lower())):
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_RTS_HEAD
                    head_flag = True
                xml_str += CE_NC_MERGE_EVPN_RTS_CONTEXT % ('export_extcommunity', ele)
                self.updates_cmd.append('    vpn-target %s export-extcommunity' % ele)
    if vpn_target_import:
        for ele in vpn_target_import:
            if ele.lower() != 'auto' and (not self.is_vpn_target_exist('import_extcommunity', ele.lower())):
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_RTS_HEAD
                    head_flag = True
                xml_str += CE_NC_MERGE_EVPN_RTS_CONTEXT % ('import_extcommunity', ele)
                self.updates_cmd.append('    vpn-target %s import-extcommunity' % ele)
    if head_flag:
        xml_str += CE_NC_MERGE_EVPN_RTS_TAIL
    xml_str += CE_NC_MERGE_EVPN_CONFIG_TAIL
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'MERGE_EVPN_BD_CONFIG')
    self.changed = True