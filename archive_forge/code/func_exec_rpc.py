from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
def exec_rpc(module, *args, **kwargs):
    connection = NetconfConnection(module._socket_path)
    return connection.execute_rpc(*args, **kwargs)