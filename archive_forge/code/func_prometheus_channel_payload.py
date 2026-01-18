from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def prometheus_channel_payload(data, payload):
    payload['type'] = 'prometheus-alertmanager'
    payload['settings']['url'] = data['prometheus_url']
    if data.get('prometheus_username'):
        payload['settings']['basicAuthUser'] = data['prometheus_username']
    if data.get('prometheus_password'):
        payload['settings']['basicAuthPassword'] = data['prometheus_password']