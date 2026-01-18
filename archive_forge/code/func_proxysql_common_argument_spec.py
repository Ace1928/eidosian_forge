from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.basic import missing_required_lib
def proxysql_common_argument_spec():
    return dict(login_user=dict(type='str', default=None), login_password=dict(type='str', no_log=True), login_host=dict(type='str', default='127.0.0.1'), login_port=dict(type='int', default=6032), login_unix_socket=dict(type='str'), config_file=dict(type='path', default=''))