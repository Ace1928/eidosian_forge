from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def handle_add_and_set_result(self, connection, version, api_url, payload, session_uid, auto_publish_session=False):
    code, response = self.handle_call(connection, version, api_url, payload, True, session_uid, auto_publish_session)
    result = {'code': code, 'response': response, 'changed': True}
    return result