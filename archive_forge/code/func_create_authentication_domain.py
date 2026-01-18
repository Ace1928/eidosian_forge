from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def create_authentication_domain(self, **kwargs):
    """ Create domain of authentication """
    domain_name = kwargs['domain_name']
    authen_scheme_name = kwargs['authen_scheme_name']
    module = kwargs['module']
    conf_str = CE_CREATE_AUTHENTICATION_DOMAIN % (domain_name, authen_scheme_name)
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Create authentication domain failed.')
    cmds = []
    cmd = 'domain %s' % domain_name
    cmds.append(cmd)
    cmd = 'authentication-scheme %s' % authen_scheme_name
    cmds.append(cmd)
    return cmds