from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_rules_amount(connection, version, show_rulebase_payload, show_rulebase_command):
    payload = {'limit': 0}
    payload.update(show_rulebase_payload)
    code, response = send_request(connection, version, show_rulebase_command, payload)
    return int(response['total'])