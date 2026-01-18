from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def delete_local_user_info(self, **kwargs):
    """ Delete local user information by netconf """
    module = kwargs['module']
    local_user_name = module.params['local_user_name']
    conf_str = CE_DELETE_LOCAL_USER_INFO_HEADER % local_user_name
    conf_str += CE_DELETE_LOCAL_USER_INFO_TAIL
    cmds = []
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Delete local user info failed.')
    cmd = 'undo local-user %s' % local_user_name
    cmds.append(cmd)
    return cmds