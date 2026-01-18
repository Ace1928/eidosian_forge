from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def get_hwtacacs_server_cfg_ipv6(self, **kwargs):
    """ Get hwtacacs server configure ipv6 """
    module = kwargs['module']
    hwtacacs_template = module.params['hwtacacs_template']
    hwtacacs_server_ipv6 = module.params['hwtacacs_server_ipv6']
    hwtacacs_server_type = module.params['hwtacacs_server_type']
    hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
    hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
    state = module.params['state']
    result = dict()
    result['hwtacacs_server_cfg_ipv6'] = []
    need_cfg = False
    conf_str = CE_GET_HWTACACS_SERVER_CFG_IPV6 % hwtacacs_template
    recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
    if '<data/>' in recv_xml:
        if state == 'present':
            need_cfg = True
    else:
        xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        hwtacacs_server_cfg_ipv6 = root.findall('hwtacacs/hwTacTempCfgs/hwTacTempCfg/hwTacIpv6SrvCfgs/hwTacIpv6SrvCfg')
        if hwtacacs_server_cfg_ipv6:
            for tmp in hwtacacs_server_cfg_ipv6:
                tmp_dict = dict()
                for site in tmp:
                    if site.tag in ['serverIpAddress', 'serverType', 'isSecondaryServer', 'vpnName']:
                        tmp_dict[site.tag] = site.text
                result['hwtacacs_server_cfg_ipv6'].append(tmp_dict)
        if result['hwtacacs_server_cfg_ipv6']:
            cfg = dict()
            config_list = list()
            if hwtacacs_server_ipv6:
                cfg['serverIpAddress'] = hwtacacs_server_ipv6.lower()
            if hwtacacs_server_type:
                cfg['serverType'] = hwtacacs_server_type.lower()
            if hwtacacs_is_secondary_server:
                cfg['isSecondaryServer'] = str(hwtacacs_is_secondary_server).lower()
            if hwtacacs_vpn_name:
                cfg['vpnName'] = hwtacacs_vpn_name.lower()
            for tmp in result['hwtacacs_server_cfg_ipv6']:
                exist_cfg = dict()
                if hwtacacs_server_ipv6:
                    exist_cfg['serverIpAddress'] = tmp.get('serverIpAddress').lower()
                if hwtacacs_server_type:
                    exist_cfg['serverType'] = tmp.get('serverType').lower()
                if hwtacacs_is_secondary_server:
                    exist_cfg['isSecondaryServer'] = tmp.get('isSecondaryServer').lower()
                if hwtacacs_vpn_name:
                    exist_cfg['vpnName'] = tmp.get('vpnName').lower()
                config_list.append(exist_cfg)
            if cfg in config_list:
                if state == 'present':
                    need_cfg = False
                else:
                    need_cfg = True
            elif state == 'present':
                need_cfg = True
            else:
                need_cfg = False
    result['need_cfg'] = need_cfg
    return result