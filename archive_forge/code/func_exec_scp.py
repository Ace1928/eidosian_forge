from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible.module_utils.connection import Connection, ConnectionError
def exec_scp(module, command):
    connection = Connection(module._socket_path)
    return connection.scp(**command)