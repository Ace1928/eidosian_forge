from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_conn_params(module, params_dict, warn_db_default=True):
    """Get connection parameters from the passed dictionary.

    Return a dictionary with parameters to connect to PostgreSQL server.

    Args:
        module (AnsibleModule) -- object of ansible.module_utils.basic.AnsibleModule class
        params_dict (dict) -- dictionary with variables

    Kwargs:
        warn_db_default (bool) -- warn that the default DB is used (default True)
    """
    params_map = {'login_host': 'host', 'login_user': 'user', 'login_password': 'password', 'port': 'port', 'ssl_mode': 'sslmode', 'ca_cert': 'sslrootcert', 'ssl_cert': 'sslcert', 'ssl_key': 'sslkey'}
    if PSYCOPG_VERSION >= LooseVersion('2.7.0'):
        if params_dict.get('db'):
            params_map['db'] = 'dbname'
        elif params_dict.get('database'):
            params_map['database'] = 'dbname'
        elif params_dict.get('login_db'):
            params_map['login_db'] = 'dbname'
        elif warn_db_default:
            module.warn('Database name has not been passed, used default database to connect to.')
    elif params_dict.get('db'):
        params_map['db'] = 'database'
    elif params_dict.get('database'):
        params_map['database'] = 'database'
    elif params_dict.get('login_db'):
        params_map['login_db'] = 'database'
    elif warn_db_default:
        module.warn('Database name has not been passed, used default database to connect to.')
    kw = dict(((params_map[k], v) for k, v in iteritems(params_dict) if k in params_map and v != '' and (v is not None)))
    is_localhost = False
    if 'host' not in kw or kw['host'] in [None, 'localhost']:
        is_localhost = True
    if is_localhost and params_dict['login_unix_socket'] != '':
        kw['host'] = params_dict['login_unix_socket']
    if params_dict.get('connect_params'):
        kw.update(params_dict['connect_params'])
    return kw