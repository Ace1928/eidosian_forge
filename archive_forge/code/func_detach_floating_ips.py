from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def detach_floating_ips(module, rest, ip):
    payload = {'type': 'unassign'}
    response = rest.post('floating_ips/{0}/actions'.format(ip), data=payload)
    status_code = response.status_code
    json_data = response.json
    if status_code == 201:
        json_data = wait_action(module, rest, ip, json_data['action']['id'])
        module.exit_json(changed=True, msg='Detached floating ip {0}'.format(ip), data=json_data)
        action = json_data.get('action', None)
        action_id = action.get('id', None)
        if action is None:
            module.fail_json(changed=False, msg='Error retrieving detach action. Got: {0}'.format(action))
        if action_id is None:
            module.fail_json(changed=False, msg='Error retrieving detach action ID. Got: {0}'.format(action_id))
    else:
        module.fail_json(changed=False, msg='Error detaching floating ip [{0}: {1}]'.format(status_code, json_data['message']))