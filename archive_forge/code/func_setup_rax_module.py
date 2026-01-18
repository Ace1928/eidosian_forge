from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def setup_rax_module(module, rax_module, region_required=True):
    """Set up pyrax in a standard way for all modules"""
    rax_module.USER_AGENT = 'ansible/%s %s' % (module.ansible_version, rax_module.USER_AGENT)
    api_key = module.params.get('api_key')
    auth_endpoint = module.params.get('auth_endpoint')
    credentials = module.params.get('credentials')
    env = module.params.get('env')
    identity_type = module.params.get('identity_type')
    region = module.params.get('region')
    tenant_id = module.params.get('tenant_id')
    tenant_name = module.params.get('tenant_name')
    username = module.params.get('username')
    verify_ssl = module.params.get('validate_certs')
    if env is not None:
        rax_module.set_environment(env)
    rax_module.set_setting('identity_type', identity_type)
    if verify_ssl is not None:
        rax_module.set_setting('verify_ssl', verify_ssl)
    if auth_endpoint is not None:
        rax_module.set_setting('auth_endpoint', auth_endpoint)
    if tenant_id is not None:
        rax_module.set_setting('tenant_id', tenant_id)
    if tenant_name is not None:
        rax_module.set_setting('tenant_name', tenant_name)
    try:
        username = username or os.environ.get('RAX_USERNAME')
        if not username:
            username = rax_module.get_setting('keyring_username')
            if username:
                api_key = 'USE_KEYRING'
        if not api_key:
            api_key = os.environ.get('RAX_API_KEY')
        credentials = credentials or os.environ.get('RAX_CREDENTIALS') or os.environ.get('RAX_CREDS_FILE')
        region = region or os.environ.get('RAX_REGION') or rax_module.get_setting('region')
    except KeyError as e:
        module.fail_json(msg='Unable to load %s' % e.message)
    try:
        if api_key and username:
            if api_key == 'USE_KEYRING':
                rax_module.keyring_auth(username, region=region)
            else:
                rax_module.set_credentials(username, api_key=api_key, region=region)
        elif credentials:
            credentials = os.path.expanduser(credentials)
            rax_module.set_credential_file(credentials, region=region)
        else:
            raise Exception('No credentials supplied!')
    except Exception as e:
        if e.message:
            msg = str(e.message)
        else:
            msg = repr(e)
        module.fail_json(msg=msg)
    if region_required and region not in rax_module.regions:
        module.fail_json(msg='%s is not a valid region, must be one of: %s' % (region, ','.join(rax_module.regions)))
    return rax_module