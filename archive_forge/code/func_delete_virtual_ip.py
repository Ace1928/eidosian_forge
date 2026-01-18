from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_virtual_ip(self):
    """delete virtual ip info"""
    if self.is_virtual_ip_exist():
        conf_str = CE_NC_DELETE_VRRP_VIRTUAL_IP_INFO % (self.vrid, self.interface, self.virtual_ip)
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: delete virtual ip info failed.')
        self.updates_cmd.append('interface %s' % self.interface)
        self.updates_cmd.append('undo vrrp vrid %s virtual-ip %s ' % (self.vrid, self.virtual_ip))
        self.changed = True