from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def merge_bgp_peer(self, **kwargs):
    """ merge_bgp_peer """
    module = kwargs['module']
    vrf_name = module.params['vrf_name']
    peer_addr = module.params['peer_addr']
    remote_as = module.params['remote_as']
    conf_str = CE_MERGE_BGP_PEER_HEADER % (vrf_name, peer_addr) + '<remoteAs>%s</remoteAs>' % remote_as + CE_MERGE_BGP_PEER_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Merge bgp peer failed.')
    cmds = []
    cmd = 'peer %s as-number %s' % (peer_addr, remote_as)
    cmds.append(cmd)
    return cmds