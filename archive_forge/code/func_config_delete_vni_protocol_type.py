from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_delete_vni_protocol_type(self, nve_name, vni_id, protocol_type):
    """remove vni protocol type"""
    if not self.is_vni_protocol_exist(nve_name, vni_id, protocol_type):
        return
    cfg_xml = CE_NC_DELETE_VNI_PROTOCOL % (nve_name, vni_id, protocol_type)
    recv_xml = set_nc_config(self.module, cfg_xml)
    self.check_response(recv_xml, 'DELETE_VNI_PEER_PROTOCOL')
    self.updates_cmd.append('interface %s' % nve_name)
    self.updates_cmd.append('undo vni %s head-end peer-list protocol bgp ' % vni_id)
    self.changed = True