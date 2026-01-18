from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def handle_publish(self, connection, version, payload):
    publish_code, publish_response = send_request(connection, version, 'publish')
    if publish_code != 200:
        self.discard_and_fail(publish_code, publish_response, connection, version)
    if payload.get('wait_for_task'):
        self.wait_for_task(version, connection, publish_response['task-id'])