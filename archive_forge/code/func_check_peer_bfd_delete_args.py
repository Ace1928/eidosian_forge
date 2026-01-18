from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def check_peer_bfd_delete_args(self, **kwargs):
    """ check_peer_bfd_delete_args """
    module = kwargs['module']
    result = dict()
    need_cfg = False
    state = module.params['state']
    if state == 'present':
        result['need_cfg'] = need_cfg
        return result
    vrf_name = module.params['vrf_name']
    if vrf_name:
        if len(vrf_name) > 31 or len(vrf_name) == 0:
            module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
    peer_addr = module.params['peer_addr']
    is_bfd_block = module.params['is_bfd_block']
    if is_bfd_block != 'no_use':
        conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isBfdBlock></isBfdBlock>' + CE_GET_PEER_BFD_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<isBfdBlock>(.*)</isBfdBlock>.*', recv_xml)
            if re_find:
                result['is_bfd_block'] = re_find
                if re_find[0] == is_bfd_block:
                    need_cfg = True
    multiplier = module.params['multiplier']
    if multiplier:
        if int(multiplier) > 50 or int(multiplier) < 3:
            module.fail_json(msg='Error: The value of multiplier %s is out of [3 - 50].' % multiplier)
        conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<multiplier></multiplier>' + CE_GET_PEER_BFD_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<multiplier>(.*)</multiplier>.*', recv_xml)
            if re_find:
                result['multiplier'] = re_find
                if re_find[0] == multiplier:
                    need_cfg = True
    is_bfd_enable = module.params['is_bfd_enable']
    if is_bfd_enable != 'no_use':
        conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isBfdEnable></isBfdEnable>' + CE_GET_PEER_BFD_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<isBfdEnable>(.*)</isBfdEnable>.*', recv_xml)
            if re_find:
                result['is_bfd_enable'] = re_find
                if re_find[0] == is_bfd_enable:
                    need_cfg = True
    rx_interval = module.params['rx_interval']
    if rx_interval:
        if int(rx_interval) > 1000 or int(rx_interval) < 50:
            module.fail_json(msg='Error: The value of rx_interval %s is out of [50 - 1000].' % rx_interval)
        conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<rxInterval></rxInterval>' + CE_GET_PEER_BFD_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<rxInterval>(.*)</rxInterval>.*', recv_xml)
            if re_find:
                result['rx_interval'] = re_find
                if re_find[0] == rx_interval:
                    need_cfg = True
    tx_interval = module.params['tx_interval']
    if tx_interval:
        if int(tx_interval) > 1000 or int(tx_interval) < 50:
            module.fail_json(msg='Error: The value of tx_interval %s is out of [50 - 1000].' % tx_interval)
        conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<txInterval></txInterval>' + CE_GET_PEER_BFD_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<txInterval>(.*)</txInterval>.*', recv_xml)
            if re_find:
                result['tx_interval'] = re_find
                if re_find[0] == tx_interval:
                    need_cfg = True
    is_single_hop = module.params['is_single_hop']
    if is_single_hop != 'no_use':
        conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isSingleHop></isSingleHop>' + CE_GET_PEER_BFD_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<isSingleHop>(.*)</isSingleHop>.*', recv_xml)
            if re_find:
                result['is_single_hop'] = re_find
                if re_find[0] == is_single_hop:
                    need_cfg = True
    result['need_cfg'] = need_cfg
    return result