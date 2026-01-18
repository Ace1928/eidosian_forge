from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_module_argument(**kwargs):
    """ Check module argument """
    module = kwargs['module']
    authen_scheme_name = module.params['authen_scheme_name']
    author_scheme_name = module.params['author_scheme_name']
    acct_scheme_name = module.params['acct_scheme_name']
    domain_name = module.params['domain_name']
    radius_server_group = module.params['radius_server_group']
    hwtacas_template = module.params['hwtacas_template']
    local_user_group = module.params['local_user_group']
    if authen_scheme_name:
        if len(authen_scheme_name) > 32:
            module.fail_json(msg='Error: authen_scheme_name %s is large than 32.' % authen_scheme_name)
        check_name(module=module, name=authen_scheme_name, invalid_char=INVALID_SCHEME_CHAR)
    if author_scheme_name:
        if len(author_scheme_name) > 32:
            module.fail_json(msg='Error: author_scheme_name %s is large than 32.' % author_scheme_name)
        check_name(module=module, name=author_scheme_name, invalid_char=INVALID_SCHEME_CHAR)
    if acct_scheme_name:
        if len(acct_scheme_name) > 32:
            module.fail_json(msg='Error: acct_scheme_name %s is large than 32.' % acct_scheme_name)
        check_name(module=module, name=acct_scheme_name, invalid_char=INVALID_SCHEME_CHAR)
    if domain_name:
        if len(domain_name) > 64:
            module.fail_json(msg='Error: domain_name %s is large than 64.' % domain_name)
        check_name(module=module, name=domain_name, invalid_char=INVALID_DOMAIN_CHAR)
        if domain_name == '-' or domain_name == '--':
            module.fail_json(msg='domain_name %s is invalid.' % domain_name)
    if radius_server_group and len(radius_server_group) > 32:
        module.fail_json(msg='Error: radius_server_group %s is large than 32.' % radius_server_group)
    if hwtacas_template and len(hwtacas_template) > 32:
        module.fail_json(msg='Error: hwtacas_template %s is large than 32.' % hwtacas_template)
    if local_user_group:
        if len(local_user_group) > 32:
            module.fail_json(msg='Error: local_user_group %s is large than 32.' % local_user_group)
        check_name(module=module, name=local_user_group, invalid_char=INVALID_GROUP_CHAR)