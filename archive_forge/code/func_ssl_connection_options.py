from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def ssl_connection_options(connection_params, module):
    connection_params['ssl'] = True
    if module.params['ssl_cert_reqs'] is not None:
        connection_params['ssl_cert_reqs'] = getattr(ssl_lib, module.params['ssl_cert_reqs'])
    connection_params = add_option_if_not_none('ssl_ca_certs', module, connection_params)
    connection_params = add_option_if_not_none('ssl_crlfile', module, connection_params)
    connection_params = add_option_if_not_none('ssl_certfile', module, connection_params)
    connection_params = add_option_if_not_none('ssl_keyfile', module, connection_params)
    connection_params = add_option_if_not_none('ssl_pem_passphrase', module, connection_params)
    if module.params['auth_mechanism'] is not None:
        connection_params['authMechanism'] = module.params['auth_mechanism']
    if module.params['connection_options'] is not None:
        for item in module.params['connection_options']:
            if isinstance(item, dict):
                for key, value in item.items():
                    connection_params[key] = value
            elif isinstance(item, str) and '=' in item:
                connection_params[item.split('=')[0]] = item.split('=')[1]
            else:
                raise ValueError('Invalid value supplied in connection_options: {0} .'.format(str(item)))
    return connection_params