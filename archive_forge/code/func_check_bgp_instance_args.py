from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_bgp_instance_args(self, **kwargs):
    """ check_bgp_instance_args """
    module = kwargs['module']
    state = module.params['state']
    need_cfg = False
    vrf_name = module.params['vrf_name']
    if vrf_name:
        if len(vrf_name) > 31 or len(vrf_name) == 0:
            module.fail_json(msg='the len of vrf_name %s is out of [1 - 31].' % vrf_name)
        conf_str = CE_GET_BGP_INSTANCE_HEADER + '<vrfName></vrfName>' + CE_GET_BGP_INSTANCE_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        check_vrf_name = vrf_name
        if state == 'present':
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<vrfName>(.*)</vrfName>.*', recv_xml)
                if re_find:
                    if check_vrf_name not in re_find:
                        need_cfg = True
                else:
                    need_cfg = True
        elif '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<vrfName>(.*)</vrfName>.*', recv_xml)
            if re_find:
                if check_vrf_name in re_find:
                    need_cfg = True
    return need_cfg