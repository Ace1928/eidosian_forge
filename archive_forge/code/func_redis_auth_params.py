from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
import traceback
def redis_auth_params(module):
    login_host = module.params['login_host']
    login_user = module.params['login_user']
    login_password = module.params['login_password']
    login_port = module.params['login_port']
    tls = module.params['tls']
    validate_certs = 'required' if module.params['validate_certs'] else None
    ca_certs = module.params['ca_certs']
    if tls and ca_certs is None:
        ca_certs = str(certifi.where())
    if tuple(map(int, redis_version.split('.'))) < (3, 4, 0) and login_user is not None:
        module.fail_json(msg='The option `username` in only supported with redis >= 3.4.0.')
    params = {'host': login_host, 'port': login_port, 'password': login_password, 'ssl_ca_certs': ca_certs, 'ssl_cert_reqs': validate_certs, 'ssl': tls}
    if login_user is not None:
        params['username'] = login_user
    return params