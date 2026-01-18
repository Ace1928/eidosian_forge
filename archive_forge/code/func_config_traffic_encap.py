from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_traffic_encap(self):
    """configure traffic encapsulation types"""
    if not self.l2sub_info:
        self.module.fail_json(msg='Error: Interface %s does not exist.' % self.l2_sub_interface)
    if not self.encapsulation:
        return
    xml_str = ''
    if self.encapsulation in ['default', 'untag']:
        if self.state == 'present':
            if self.encapsulation != self.l2sub_info.get('flowType'):
                xml_str = CE_NC_SET_ENCAP % (self.l2_sub_interface, self.encapsulation)
                self.updates_cmd.append('interface %s' % self.l2_sub_interface)
                self.updates_cmd.append('encapsulation %s' % self.encapsulation)
        elif self.encapsulation == self.l2sub_info.get('flowType'):
            xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
            self.updates_cmd.append('interface %s' % self.l2_sub_interface)
            self.updates_cmd.append('undo encapsulation %s' % self.encapsulation)
    elif self.encapsulation == 'none':
        if self.state == 'present':
            if self.encapsulation != self.l2sub_info.get('flowType'):
                xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                self.updates_cmd.append('interface %s' % self.l2_sub_interface)
                self.updates_cmd.append('undo encapsulation %s' % self.l2sub_info.get('flowType'))
    elif self.encapsulation == 'dot1q':
        self.config_traffic_encap_dot1q()
        return
    elif self.encapsulation == 'qinq':
        self.config_traffic_encap_qinq()
        return
    else:
        pass
    if not xml_str:
        return
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'CONFIG_INTF_ENCAP')
    self.changed = True