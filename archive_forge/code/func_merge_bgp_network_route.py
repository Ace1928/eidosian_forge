from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def merge_bgp_network_route(self, **kwargs):
    """ merge_bgp_network_route """
    module = kwargs['module']
    vrf_name = module.params['vrf_name']
    af_type = module.params['af_type']
    network_address = module.params['network_address']
    mask_len = module.params['mask_len']
    conf_str = CE_MERGE_BGP_NETWORK_ROUTE_HEADER % (vrf_name, af_type, network_address, mask_len) + CE_MERGE_BGP_NETWORK_ROUTE_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Merge bgp network route failed.')
    cmds = []
    cmd = 'network %s %s' % (network_address, mask_len)
    cmds.append(cmd)
    return cmds