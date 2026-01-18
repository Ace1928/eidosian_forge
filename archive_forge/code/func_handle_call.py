from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def handle_call(self, connection, version, api_url, payload, to_discard_on_failure, session_uid=None, to_publish=False):
    code, response = send_request(connection, version, api_url, payload)
    if code != 200:
        if to_discard_on_failure:
            self.discard_and_fail(code, response, connection, version, session_uid)
        elif 'object_not_found' not in response.get('code') and 'not found' not in response.get('message'):
            raise _fail_json('Checkpoint session with ID: {0}'.format(session_uid) + ', ' + parse_fail_message(code, response))
    elif 'wait_for_task' in payload and payload['wait_for_task']:
        if 'task-id' in response:
            response = self.wait_for_task(version, connection, response['task-id'])
        elif 'tasks' in response:
            for task in response['tasks']:
                if 'task-id' in task:
                    task_id = task['task-id']
                    response[task_id] = self.wait_for_task(version, connection, task['task-id'])
            del response['tasks']
    if to_publish:
        self.handle_publish(connection, version, payload)
    return (code, response)