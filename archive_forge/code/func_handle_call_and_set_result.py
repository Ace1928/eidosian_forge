from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def handle_call_and_set_result(connection, version, call, payload, module, result):
    response = handle_call(connection, version, call, payload, module, True, True)
    result['changed'] = True
    result[call] = response