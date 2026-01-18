from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def delete_peer_bfd(self, **kwargs):
    """ delete_peer_bfd """
    module = kwargs['module']
    vrf_name = module.params['vrf_name']
    peer_addr = module.params['peer_addr']
    conf_str = CE_DELETE_PEER_BFD_HEADER % (vrf_name, peer_addr)
    cmds = []
    is_bfd_block = module.params['is_bfd_block']
    if is_bfd_block != 'no_use':
        conf_str += '<isBfdBlock>%s</isBfdBlock>' % is_bfd_block
        cmd = 'undo peer %s bfd block' % peer_addr
        cmds.append(cmd)
    multiplier = module.params['multiplier']
    if multiplier:
        conf_str += '<multiplier>%s</multiplier>' % multiplier
        cmd = 'undo peer %s bfd detect-multiplier %s' % (peer_addr, multiplier)
        cmds.append(cmd)
    is_bfd_enable = module.params['is_bfd_enable']
    if is_bfd_enable != 'no_use':
        conf_str += '<isBfdEnable>%s</isBfdEnable>' % is_bfd_enable
        cmd = 'undo peer %s bfd enable' % peer_addr
        cmds.append(cmd)
    rx_interval = module.params['rx_interval']
    if rx_interval:
        conf_str += '<rxInterval>%s</rxInterval>' % rx_interval
        cmd = 'undo peer %s bfd min-rx-interval %s' % (peer_addr, rx_interval)
        cmds.append(cmd)
    tx_interval = module.params['tx_interval']
    if tx_interval:
        conf_str += '<txInterval>%s</txInterval>' % tx_interval
        cmd = 'undo peer %s bfd min-tx-interval %s' % (peer_addr, tx_interval)
        cmds.append(cmd)
    is_single_hop = module.params['is_single_hop']
    if is_single_hop != 'no_use':
        conf_str += '<isSingleHop>%s</isSingleHop>' % is_single_hop
        cmd = 'undo peer %s bfd enable single-hop-prefer' % peer_addr
        cmds.append(cmd)
    conf_str += CE_DELETE_PEER_BFD_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Delete peer bfd failed.')
    return cmds