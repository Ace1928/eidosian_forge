from __future__ import absolute_import, division, print_function
import re
import socket
import sys
import traceback
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import exec_command, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import NetconfConnection
def get_nc_connection(module):
    global _DEVICE_NC_CONNECTION
    if not _DEVICE_NC_CONNECTION:
        load_params(module)
        conn = NetconfConnection(module._socket_path)
        _DEVICE_NC_CONNECTION = conn
    return _DEVICE_NC_CONNECTION