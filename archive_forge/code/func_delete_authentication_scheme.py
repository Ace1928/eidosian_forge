from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_authentication_scheme(self, **kwargs):
    """ Delete scheme of authentication """
    authen_scheme_name = kwargs['authen_scheme_name']
    first_authen_mode = kwargs['first_authen_mode']
    module = kwargs['module']
    if authen_scheme_name == 'default':
        return SUCCESS
    conf_str = CE_DELETE_AUTHENTICATION_SCHEME % (authen_scheme_name, first_authen_mode)
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Delete authentication scheme failed.')
    cmds = []
    cmd = 'undo authentication-scheme %s' % authen_scheme_name
    cmds.append(cmd)
    cmd = 'authentication-mode none'
    cmds.append(cmd)
    return cmds