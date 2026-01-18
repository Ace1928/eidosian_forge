from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_vap_sub_intf(self):
    """configure a Layer 2 sub-interface as a service access point"""
    if not self.vap_info:
        self.module.fail_json(msg='Error: Bridge domain %s does not exist.' % self.bridge_domain_id)
    xml_str = ''
    if self.state == 'present':
        if self.l2_sub_interface not in self.vap_info['intfList']:
            self.updates_cmd.append('interface %s' % self.l2_sub_interface)
            self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
            xml_str = CE_NC_MERGE_BD_INTF % (self.bridge_domain_id, self.l2_sub_interface)
    elif self.l2_sub_interface in self.vap_info['intfList']:
        self.updates_cmd.append('interface %s' % self.l2_sub_interface)
        self.updates_cmd.append('undo bridge-domain %s' % self.bridge_domain_id)
        xml_str = CE_NC_DELETE_BD_INTF % (self.bridge_domain_id, self.l2_sub_interface)
    if not xml_str:
        return
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'CONFIG_VAP_SUB_INTERFACE')
    self.changed = True