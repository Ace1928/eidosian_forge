from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_delete_vni_peer_ip(self, nve_name, vni_id, peer_ip_list):
    """remove vni peer ip"""
    for peer_ip in peer_ip_list:
        if not self.is_vni_peer_list_exist(nve_name, vni_id, peer_ip):
            self.module.fail_json(msg='Error: The %s does not exist' % peer_ip)
    config = False
    nve_peer_info = list()
    for nve_peer in self.nve_info['vni_peer_ips']:
        if nve_peer['vniId'] == vni_id:
            nve_peer_info = nve_peer.get('peerAddr')
    for peer in nve_peer_info:
        if peer not in peer_ip_list:
            config = True
    if not config:
        cfg_xml = CE_NC_DELETE_VNI_PEER_ADDRESS_IP_HEAD % (nve_name, vni_id)
        for peer_ip in peer_ip_list:
            cfg_xml += CE_NC_DELETE_VNI_PEER_ADDRESS_IP_DELETE % peer_ip
        cfg_xml += CE_NC_DELETE_VNI_PEER_ADDRESS_IP_END
    else:
        cfg_xml = CE_NC_DELETE_PEER_ADDRESS_IP_HEAD % (nve_name, vni_id)
        for peer_ip in peer_ip_list:
            cfg_xml += CE_NC_DELETE_VNI_PEER_ADDRESS_IP_DELETE % peer_ip
        cfg_xml += CE_NC_DELETE_PEER_ADDRESS_IP_END
    recv_xml = set_nc_config(self.module, cfg_xml)
    self.check_response(recv_xml, 'DELETE_VNI_PEER_IP')
    self.updates_cmd.append('interface %s' % nve_name)
    for peer_ip in peer_ip_list:
        cmd_output = 'undo vni %s head-end peer-list %s' % (vni_id, peer_ip)
        self.updates_cmd.append(cmd_output)
    self.changed = True