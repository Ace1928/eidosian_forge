from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def check_bgp_import_network_route(self, **kwargs):
    """ check_bgp_import_network_route """
    module = kwargs['module']
    result = dict()
    import_need_cfg = False
    network_need_cfg = False
    vrf_name = module.params['vrf_name']
    state = module.params['state']
    af_type = module.params['af_type']
    import_protocol = module.params['import_protocol']
    import_process_id = module.params['import_process_id']
    if import_protocol and (import_protocol != 'direct' and import_protocol != 'static'):
        if not import_process_id:
            module.fail_json(msg='Error: Please input import_protocol and import_process_id value at the same time.')
        elif int(import_process_id) < 0:
            module.fail_json(msg='Error: The value of import_process_id %s is out of [0 - 4294967295].' % import_process_id)
    if import_process_id:
        if not import_protocol:
            module.fail_json(msg='Error: Please input import_protocol and import_process_id value at the same time.')
    network_address = module.params['network_address']
    mask_len = module.params['mask_len']
    if network_address:
        if not mask_len:
            module.fail_json(msg='Error: Please input network_address and mask_len value at the same time.')
    if mask_len:
        if not network_address:
            module.fail_json(msg='Error: Please input network_address and mask_len value at the same time.')
    conf_str = CE_GET_BGP_IMPORT_AND_NETWORK_ROUTE % (vrf_name, af_type)
    recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
    if import_protocol:
        if import_protocol == 'direct' or import_protocol == 'static':
            import_process_id = '0'
        elif not import_process_id or import_process_id == '0':
            module.fail_json(msg='Error: Please input import_process_id not 0 when import_protocol is [ospf, isis, rip, ospfv3, ripng].')
        bgp_import_route_new = (import_protocol, import_process_id)
        if state == 'present':
            if '<data/>' in recv_xml:
                import_need_cfg = True
            else:
                re_find = re.findall('.*<importProtocol>(.*)</importProtocol>.*\\s.*<importProcessId>(.*)</importProcessId>.*', recv_xml)
                if re_find:
                    result['bgp_import_route'] = re_find
                    result['vrf_name'] = vrf_name
                    if bgp_import_route_new not in re_find:
                        import_need_cfg = True
                else:
                    import_need_cfg = True
        elif '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<importProtocol>(.*)</importProtocol>.*\\s.*<importProcessId>(.*)</importProcessId>.*', recv_xml)
            if re_find:
                result['bgp_import_route'] = re_find
                result['vrf_name'] = vrf_name
                if bgp_import_route_new in re_find:
                    import_need_cfg = True
    if network_address and mask_len:
        bgp_network_route_new = (network_address, mask_len)
        if not check_ip_addr(ipaddr=network_address):
            module.fail_json(msg='Error: The network_address %s is invalid.' % network_address)
        if len(mask_len) > 128:
            module.fail_json(msg='Error: The len of mask_len %s is out of [0 - 128].' % mask_len)
        if state == 'present':
            if '<data/>' in recv_xml:
                network_need_cfg = True
            else:
                re_find = re.findall('.*<networkAddress>(.*)</networkAddress>.*\\s.*<maskLen>(.*)</maskLen>.*', recv_xml)
                if re_find:
                    result['bgp_network_route'] = re_find
                    result['vrf_name'] = vrf_name
                    if bgp_network_route_new not in re_find:
                        network_need_cfg = True
                else:
                    network_need_cfg = True
        elif '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<networkAddress>(.*)</networkAddress>.*\\s.*<maskLen>(.*)</maskLen>.*', recv_xml)
            if re_find:
                result['bgp_network_route'] = re_find
                result['vrf_name'] = vrf_name
                if bgp_network_route_new in re_find:
                    network_need_cfg = True
    result['import_need_cfg'] = import_need_cfg
    result['network_need_cfg'] = network_need_cfg
    return result