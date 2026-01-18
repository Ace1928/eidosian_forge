from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def delete_radius_server_name(self, **kwargs):
    """ Delete radius server name """
    module = kwargs['module']
    radius_group_name = module.params['radius_group_name']
    radius_server_type = module.params['radius_server_type']
    radius_server_name = module.params['radius_server_name']
    radius_server_port = module.params['radius_server_port']
    radius_server_mode = module.params['radius_server_mode']
    radius_vpn_name = module.params['radius_vpn_name']
    conf_str = CE_DELETE_RADIUS_SERVER_NAME % (radius_group_name, radius_server_type, radius_server_name, radius_server_port, radius_server_mode, radius_vpn_name)
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: delete radius server name failed.')
    cmds = []
    cmd = 'radius server group %s' % radius_group_name
    cmds.append(cmd)
    if radius_server_type == 'Authentication':
        cmd = 'undo radius server authentication hostname %s %s' % (radius_server_name, radius_server_port)
        if radius_vpn_name and radius_vpn_name != '_public_':
            cmd += ' vpn-instance %s' % radius_vpn_name
        if radius_server_mode == 'Secondary-server':
            cmd += ' secondary'
    else:
        cmd = 'undo radius server accounting hostname %s %s' % (radius_server_name, radius_server_port)
        if radius_vpn_name and radius_vpn_name != '_public_':
            cmd += ' vpn-instance %s' % radius_vpn_name
        if radius_server_mode == 'Secondary-server':
            cmd += ' secondary'
    cmds.append(cmd)
    return cmds