from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def is_equals_with_position_param(payload, connection, version, api_call_object):
    position_number, section_according_to_position = get_number_and_section_from_position(payload, connection, version, api_call_object)
    if position_number is None:
        return True
    rulebase_payload = build_rulebase_payload(api_call_object, payload, position_number)
    rulebase_command = build_rulebase_command(api_call_object)
    code, response = send_request(connection, version, rulebase_command, rulebase_payload)
    rule, section = extract_rule_and_section_from_rulebase_response(response)
    if rule['name'] == payload['name'] and section_according_to_position == section:
        return True
    else:
        return False