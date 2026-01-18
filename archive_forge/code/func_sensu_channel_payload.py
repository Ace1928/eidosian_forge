from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def sensu_channel_payload(data, payload):
    payload['settings']['url'] = data['sensu_url']
    if data.get('sensu_source'):
        payload['settings']['source'] = data['sensu_source']
    if data.get('sensu_handler'):
        payload['settings']['handler'] = data['sensu_handler']
    if data.get('sensu_username'):
        payload['settings']['username'] = data['sensu_username']
    if data.get('sensu_password'):
        payload['settings']['password'] = data['sensu_password']