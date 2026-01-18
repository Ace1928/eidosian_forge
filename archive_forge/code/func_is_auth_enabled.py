from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def is_auth_enabled(module):
    """
    Returns True if auth is enabled on the mongo instance
    For PyMongo 4+ we have to connect directly to the instance
    rather than the replicaset
    """
    auth_is_enabled = None
    connection_params = {}
    connection_params['host'] = module.params['login_host']
    connection_params['port'] = module.params['login_port']
    connection_params['directConnection'] = True
    if int(PyMongoVersion[0]) >= 4:
        connection_params['directConnection'] = True
    elif 'replica_set' in module.params and module.params['replica_set'] is not None:
        connection_params['replicaset'] = module.params['replica_set']
    if module.params['ssl']:
        connection_params = ssl_connection_options(connection_params, module)
        connection_params = rename_ssl_option_for_pymongo4(connection_params)
    try:
        myclient = MongoClient(**connection_params)
        myclient['admin'].command('listDatabases', 1.0)
        auth_is_enabled = False
    except Exception as excep:
        if hasattr(excep, 'code') and excep.code in [13]:
            auth_is_enabled = True
        if auth_is_enabled is None:
            module.fail_json(msg='Unable to determine if auth is enabled: {0}'.format(traceback.format_exc()))
    finally:
        myclient.close()
    return auth_is_enabled