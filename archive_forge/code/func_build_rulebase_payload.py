from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def build_rulebase_payload(api_call_object, payload, position_number):
    show_rulebase_required_identifiers_payload = get_relevant_show_rulebase_identifier_payload(api_call_object, payload)
    show_rulebase_required_identifiers_payload.update({'offset': position_number - 1, 'limit': 1})
    return show_rulebase_required_identifiers_payload