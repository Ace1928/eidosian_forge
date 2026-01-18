from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def grafana_notification_channel_payload(data):
    payload = {'uid': data['uid'], 'name': data['name'], 'type': data['type'], 'isDefault': data['is_default'], 'disableResolveMessage': data['disable_resolve_message'], 'settings': {'uploadImage': data['include_image']}}
    if data.get('reminder_frequency'):
        payload['sendReminder'] = True
        payload['frequency'] = data['reminder_frequency']
    if data['type'] == 'dingding':
        dingding_channel_payload(data, payload)
    elif data['type'] == 'discord':
        discord_channel_payload(data, payload)
    elif data['type'] == 'email':
        email_channel_payload(data, payload)
    elif data['type'] == 'googlechat':
        payload['settings']['url'] = data['googlechat_url']
    elif data['type'] == 'hipchat':
        hipchat_channel_payload(data, payload)
    elif data['type'] == 'kafka':
        payload['settings']['kafkaRestProxy'] = data['kafka_url']
        payload['settings']['kafkaTopic'] = data['kafka_topic']
    elif data['type'] == 'line':
        payload['settings']['token'] = data['line_token']
    elif data['type'] == 'teams':
        payload['settings']['url'] = data['teams_url']
    elif data['type'] == 'opsgenie':
        payload['settings']['apiUrl'] = data['opsgenie_url']
        payload['settings']['apiKey'] = data['opsgenie_api_key']
    elif data['type'] == 'pagerduty':
        pagerduty_channel_payload(data, payload)
    elif data['type'] == 'prometheus':
        prometheus_channel_payload(data, payload)
    elif data['type'] == 'pushover':
        pushover_channel_payload(data, payload)
    elif data['type'] == 'sensu':
        sensu_channel_payload(data, payload)
    elif data['type'] == 'slack':
        slack_channel_payload(data, payload)
    elif data['type'] == 'telegram':
        payload['settings']['bottoken'] = data['telegram_bot_token']
        payload['settings']['chatid'] = data['telegram_chat_id']
    elif data['type'] == 'threema':
        payload['settings']['gateway_id'] = data['threema_gateway_id']
        payload['settings']['recipient_id'] = data['threema_recipient_id']
        payload['settings']['api_secret'] = data['threema_api_secret']
    elif data['type'] == 'victorops':
        payload['settings']['url'] = data['victorops_url']
        if data.get('victorops_auto_resolve'):
            payload['settings']['autoResolve'] = data['victorops_auto_resolve']
    elif data['type'] == 'webhook':
        webhook_channel_payload(data, payload)
    return payload