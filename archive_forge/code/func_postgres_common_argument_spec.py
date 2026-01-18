from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def postgres_common_argument_spec():
    """
    Return a dictionary with connection options.

    The options are commonly used by most of PostgreSQL modules.
    """
    env_vars = environ
    return dict(login_user=dict(default='postgres' if not env_vars.get('PGUSER') else env_vars.get('PGUSER'), aliases=['login']), login_password=dict(default='', no_log=True), login_host=dict(default='', aliases=['host']), login_unix_socket=dict(default='', aliases=['unix_socket']), port=dict(type='int', default=int(env_vars.get('PGPORT', 5432)), aliases=['login_port']), ssl_mode=dict(default='prefer', choices=['allow', 'disable', 'prefer', 'require', 'verify-ca', 'verify-full']), ca_cert=dict(aliases=['ssl_rootcert']), ssl_cert=dict(type='path'), ssl_key=dict(type='path'), connect_params=dict(default={}, type='dict'))