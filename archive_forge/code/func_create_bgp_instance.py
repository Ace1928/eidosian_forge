from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def create_bgp_instance(self, **kwargs):
    """ create_bgp_instance """
    module = kwargs['module']
    conf_str = CE_CREATE_BGP_INSTANCE_HEADER
    cmds = []
    vrf_name = module.params['vrf_name']
    if vrf_name:
        if vrf_name == '_public_':
            return cmds
        conf_str += '<vrfName>%s</vrfName>' % vrf_name
    conf_str += CE_CREATE_BGP_INSTANCE_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Create bgp instance failed.')
    if vrf_name != '_public_':
        cmd = 'ipv4-family vpn-instance %s' % vrf_name
        cmds.append(cmd)
    return cmds