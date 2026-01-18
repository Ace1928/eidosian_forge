from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def merge_hwtacacs_server_cfg_ipv4(self, **kwargs):
    """ Merge hwtacacs server configure ipv4 """
    module = kwargs['module']
    hwtacacs_template = module.params['hwtacacs_template']
    hwtacacs_server_ip = module.params['hwtacacs_server_ip']
    hwtacacs_server_type = module.params['hwtacacs_server_type']
    hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
    hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
    hwtacacs_is_public_net = module.params['hwtacacs_is_public_net']
    conf_str = CE_MERGE_HWTACACS_SERVER_CFG_IPV4 % (hwtacacs_template, hwtacacs_server_ip, hwtacacs_server_type, str(hwtacacs_is_secondary_server).lower(), hwtacacs_vpn_name, str(hwtacacs_is_public_net).lower())
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Merge hwtacacs server config ipv4 failed.')
    cmds = []
    cmd = 'hwtacacs server template %s' % hwtacacs_template
    cmds.append(cmd)
    if hwtacacs_server_type == 'Authentication':
        cmd = 'hwtacacs server authentication %s' % hwtacacs_server_ip
        if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
            cmd += ' vpn-instance %s' % hwtacacs_vpn_name
        if hwtacacs_is_public_net:
            cmd += ' public-net'
        if hwtacacs_is_secondary_server:
            cmd += ' secondary'
    elif hwtacacs_server_type == 'Authorization':
        cmd = 'hwtacacs server authorization %s' % hwtacacs_server_ip
        if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
            cmd += ' vpn-instance %s' % hwtacacs_vpn_name
        if hwtacacs_is_public_net:
            cmd += ' public-net'
        if hwtacacs_is_secondary_server:
            cmd += ' secondary'
    elif hwtacacs_server_type == 'Accounting':
        cmd = 'hwtacacs server accounting %s' % hwtacacs_server_ip
        if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
            cmd += ' vpn-instance %s' % hwtacacs_vpn_name
        if hwtacacs_is_public_net:
            cmd += ' public-net'
        if hwtacacs_is_secondary_server:
            cmd += ' secondary'
    elif hwtacacs_server_type == 'Common':
        cmd = 'hwtacacs server %s' % hwtacacs_server_ip
        if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
            cmd += ' vpn-instance %s' % hwtacacs_vpn_name
        if hwtacacs_is_public_net:
            cmd += ' public-net'
        if hwtacacs_is_secondary_server:
            cmd += ' secondary'
    cmds.append(cmd)
    return cmds