from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def api_call_facts_for_rule(module, api_call_object, api_call_object_plural_version):
    payload = get_payload_from_parameters(module.params)
    connection = Connection(module._socket_path)
    version = get_version(module)
    if call_is_plural(api_call_object, payload):
        api_call_object = api_call_object_plural_version
    response = handle_call(connection, version, 'show-' + api_call_object, payload, module, False, False)
    result = {api_call_object: response}
    return result