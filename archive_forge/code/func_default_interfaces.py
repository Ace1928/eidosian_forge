from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def default_interfaces(self, iftype):
    """ Set interface config to default by type."""
    change = False
    xmlstr = ''
    intfs_list = self.intfs_info.get(iftype.lower())
    if not intfs_list:
        return
    for intf in intfs_list:
        if_change = False
        self.updates_cmd.append('interface %s' % intf['ifName'])
        if intf['ifDescr']:
            xmlstr += CE_NC_XML_MERGE_INTF_DES % (intf['ifName'], '')
            self.updates_cmd.append('undo description')
            if_change = True
        if is_admin_state_enable(self.intf_type) and intf['ifAdminStatus'] != 'up':
            xmlstr += CE_NC_XML_MERGE_INTF_STATUS % (intf['ifName'], 'up')
            self.updates_cmd.append('undo shutdown')
            if_change = True
        if is_portswitch_enalbe(self.intf_type) and intf['isL2SwitchPort'] != 'true':
            xmlstr += CE_NC_XML_MERGE_INTF_L2ENABLE % (intf['ifName'], 'enable')
            self.updates_cmd.append('portswitch')
            if_change = True
        if if_change:
            change = True
        else:
            self.updates_cmd.pop()
    if not change:
        return
    conf_str = '<config> ' + xmlstr + ' </config>'
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'SET_INTFS_DEFAULT')
    self.changed = True