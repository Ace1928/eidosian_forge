from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def pushover_channel_payload(data, payload):
    payload['settings']['apiToken'] = data['pushover_api_token']
    payload['settings']['userKey'] = data['pushover_user_key']
    if data.get('pushover_devices'):
        payload['settings']['device'] = ';'.join(data['pushover_devices'])
    if data.get('pushover_priority'):
        payload['settings']['priority'] = {'emergency': '2', 'high': '1', 'normal': '0', 'low': '-1', 'lowest': '-2'}[data['pushover_priority']]
    if data.get('pushover_retry'):
        payload['settings']['retry'] = str(data['pushover_retry'])
    if data.get('pushover_expire'):
        payload['settings']['expire'] = str(data['pushover_expire'])
    if data.get('pushover_alert_sound'):
        payload['settings']['sound'] = data['pushover_alert_sound']
    if data.get('pushover_ok_sound'):
        payload['settings']['okSound'] = data['pushover_ok_sound']