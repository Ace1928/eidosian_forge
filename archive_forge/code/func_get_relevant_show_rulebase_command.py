from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_relevant_show_rulebase_command(api_call_object):
    if api_call_object == 'access-rule':
        return 'show-access-rulebase'
    elif api_call_object == 'threat-rule':
        return 'show-threat-rulebase'
    elif api_call_object == 'threat-exception':
        return 'show-threat-rule-exception-rulebase'
    elif api_call_object == 'nat-rule':
        return 'show-nat-rulebase'