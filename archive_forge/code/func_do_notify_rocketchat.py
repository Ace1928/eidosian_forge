from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def do_notify_rocketchat(module, domain, token, protocol, payload):
    if token.count('/') < 1:
        module.fail_json(msg='Invalid Token specified, provide a valid token')
    rocketchat_incoming_webhook = ROCKETCHAT_INCOMING_WEBHOOK % (protocol, domain, token)
    response, info = fetch_url(module, rocketchat_incoming_webhook, data=payload)
    if info['status'] != 200:
        module.fail_json(msg='failed to send message, return status=%s' % str(info['status']))