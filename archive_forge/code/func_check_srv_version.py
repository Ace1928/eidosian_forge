from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def check_srv_version(module, client):
    srv_version = None
    try:
        srv_version = client.server_info()['version']
    except Exception as excep:
        module.fail_json(msg='Unable to get MongoDB server version: %s' % to_native(excep))
    return srv_version