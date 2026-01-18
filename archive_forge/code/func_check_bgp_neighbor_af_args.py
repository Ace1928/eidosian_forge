from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def check_bgp_neighbor_af_args(self, **kwargs):
    """ check_bgp_neighbor_af_args """
    module = kwargs['module']
    result = dict()
    need_cfg = False
    vrf_name = module.params['vrf_name']
    if vrf_name:
        if len(vrf_name) > 31 or len(vrf_name) == 0:
            module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
    state = module.params['state']
    af_type = module.params['af_type']
    remote_address = module.params['remote_address']
    if not check_ip_addr(ipaddr=remote_address):
        module.fail_json(msg='Error: The remote_address %s is invalid.' % remote_address)
    conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + CE_GET_BGP_PEER_AF_TAIL
    recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
    if state == 'present':
        if '<data/>' in recv_xml:
            need_cfg = True
        else:
            re_find = re.findall('.*<remoteAddress>(.*)</remoteAddress>.*', recv_xml)
            if re_find:
                result['remote_address'] = re_find
                result['vrf_name'] = vrf_name
                result['af_type'] = af_type
                if remote_address not in re_find:
                    need_cfg = True
            else:
                need_cfg = True
    elif '<data/>' in recv_xml:
        pass
    else:
        re_find = re.findall('.*<remoteAddress>(.*)</remoteAddress>.*', recv_xml)
        if re_find:
            result['remote_address'] = re_find
            result['vrf_name'] = vrf_name
            result['af_type'] = af_type
            if re_find[0] == remote_address:
                need_cfg = True
    result['need_cfg'] = need_cfg
    return result