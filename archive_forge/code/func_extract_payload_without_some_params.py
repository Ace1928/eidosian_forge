from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def extract_payload_without_some_params(payload, params_to_remove):
    copy_payload = dict(payload)
    for param in params_to_remove:
        if param in copy_payload:
            del copy_payload[param]
    return copy_payload