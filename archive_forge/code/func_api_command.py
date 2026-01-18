from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def api_command(module, command):
    payload = get_payload_from_parameters(module.params)
    connection = Connection(module._socket_path)
    version = get_version(module)
    code, response = send_request(connection, version, command, payload)
    result = {'changed': True}
    if command.startswith('show'):
        result['changed'] = False
    if code == 200:
        if module.params['wait_for_task']:
            if 'task-id' in response:
                response = wait_for_task(module, version, connection, response['task-id'])
            elif 'tasks' in response:
                for task in response['tasks']:
                    if 'task-id' in task:
                        task_id = task['task-id']
                        response[task_id] = wait_for_task(module, version, connection, task['task-id'])
                del response['tasks']
        result[command] = response
        handle_publish(module, connection, version)
    else:
        discard_and_fail(module, code, response, connection, version)
    return result