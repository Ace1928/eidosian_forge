from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def operate_vrf(self):
    """config/delete vrf"""
    if not self.changed:
        return
    if self.state == 'present':
        if self.description is None:
            configxmlstr = CE_NC_CREATE_VRF % (self.vrf, '')
        else:
            configxmlstr = CE_NC_CREATE_VRF % (self.vrf, self.description)
    else:
        configxmlstr = CE_NC_DELETE_VRF % (self.vrf, self.description)
    conf_str = build_config_xml(configxmlstr)
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'OPERATE_VRF')