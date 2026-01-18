from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def create_authentication_scheme(self, **kwargs):
    """ Create scheme of authentication """
    authen_scheme_name = kwargs['authen_scheme_name']
    first_authen_mode = kwargs['first_authen_mode']
    module = kwargs['module']
    conf_str = CE_CREATE_AUTHENTICATION_SCHEME % (authen_scheme_name, first_authen_mode)
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Create authentication scheme failed.')
    cmds = []
    cmd = 'authentication-scheme %s' % authen_scheme_name
    cmds.append(cmd)
    cmd = 'authentication-mode %s' % first_authen_mode
    cmds.append(cmd)
    return cmds