from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def mongo_auth(module, client, directConnection=False):
    """
    TODO: This function was extracted from code from the mongodb_replicaset module.
    We should refactor other modules to use this where appropriate. - DONE?
    @module - The calling Ansible module
    @client - The MongoDB connection object
    """
    login_user = module.params['login_user']
    login_password = module.params['login_password']
    login_database = module.params['login_database']
    atlas_auth = module.params['atlas_auth']
    fail_msg = None
    crypt_flag = 'ssl'
    if 'tls' in module.params:
        crypt_flag = 'tls'
    if not atlas_auth:
        if login_user is None and login_password is None:
            mongocnf_creds = load_mongocnf()
            if mongocnf_creds is not False:
                login_user = mongocnf_creds['user']
                login_password = mongocnf_creds['password']
        elif not all([login_user, login_password]) and module.params[crypt_flag] is False:
            fail_msg = "When supplying login arguments, both 'login_user' and 'login_password' must be provided"
        if 'create_for_localhost_exception' not in module.params and fail_msg is None:
            try:
                if is_auth_enabled(module):
                    if login_user is not None and login_password is not None:
                        client = get_mongodb_client(module, login_user, login_password, login_database, directConnection=directConnection)
                    else:
                        fail_msg = 'No credentials to authenticate'
            except Exception as excep:
                fail_msg = 'unable to connect to database: %s' % to_native(excep)
            if fail_msg is None:
                srv_version = check_srv_version(module, client)
                check_driver_compatibility(module, client, srv_version)
        elif fail_msg is None:
            if login_user is not None and login_password is not None:
                client = get_mongodb_client(module, login_user, login_password, login_database, directConnection=directConnection)
                srv_version = check_srv_version(module, client)
                check_driver_compatibility(module, client, srv_version)
            elif module.params['strict_compatibility'] is False:
                if module.params['database'] not in ['admin', '$external']:
                    fail_msg = 'The localhost login exception only allows the first admin account to be created'
        if fail_msg:
            module.fail_json(msg=fail_msg)
    else:
        if 'create_for_localhost_exception' not in module.params and fail_msg is None:
            try:
                if login_user is not None and login_password is not None:
                    client = get_mongodb_client(module, login_user, login_password, login_database)
                else:
                    fail_msg = 'No credentials to authenticate'
            except Exception as excep:
                fail_msg = 'unable to connect to database: %s' % to_native(excep)
        elif fail_msg is None:
            if login_user is not None and login_password is not None:
                client = get_mongodb_client(module, login_user, login_password, login_database, directConnection=False)
                srv_version = check_srv_version(module, client)
                check_driver_compatibility(module, client, srv_version)
            elif module.params['strict_compatibility'] is False:
                if module.params['database'] not in ['admin', '$external']:
                    fail_msg = 'The localhost login exception only allows the first admin account to be created'
        if fail_msg:
            module.fail_json(msg=fail_msg)
    return client