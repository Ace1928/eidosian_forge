from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def do_notify_slack(module, domain, token, payload):
    use_webapi = False
    if token.count('/') >= 2:
        slack_uri = SLACK_INCOMING_WEBHOOK % token
    elif re.match('^xox[abp]-\\S+$', token):
        slack_uri = SLACK_UPDATEMESSAGE_WEBAPI if 'ts' in payload else SLACK_POSTMESSAGE_WEBAPI
        use_webapi = True
    else:
        if not domain:
            module.fail_json(msg='Slack has updated its webhook API.  You need to specify a token of the form XXXX/YYYY/ZZZZ in your playbook')
        slack_uri = OLD_SLACK_INCOMING_WEBHOOK % (domain, token)
    headers = {'Content-Type': 'application/json; charset=UTF-8', 'Accept': 'application/json'}
    if use_webapi:
        headers['Authorization'] = 'Bearer ' + token
    data = module.jsonify(payload)
    response, info = fetch_url(module=module, url=slack_uri, headers=headers, method='POST', data=data)
    if info['status'] != 200:
        if use_webapi:
            obscured_incoming_webhook = slack_uri
        else:
            obscured_incoming_webhook = SLACK_INCOMING_WEBHOOK % '[obscured]'
        module.fail_json(msg=' failed to send %s to %s: %s' % (data, obscured_incoming_webhook, info['msg']))
    if use_webapi:
        return module.from_json(response.read())
    else:
        return {'webhook': 'ok'}