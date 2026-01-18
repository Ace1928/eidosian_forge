from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def get_slack_message(module, token, channel, ts):
    headers = {'Content-Type': 'application/json; charset=UTF-8', 'Accept': 'application/json', 'Authorization': 'Bearer ' + token}
    qs = urlencode({'channel': channel, 'ts': ts, 'limit': 1, 'inclusive': 'true'})
    url = SLACK_CONVERSATIONS_HISTORY_WEBAPI + '?' + qs
    response, info = fetch_url(module=module, url=url, headers=headers, method='GET')
    if info['status'] != 200:
        module.fail_json(msg='failed to get slack message')
    data = module.from_json(response.read())
    if len(data['messages']) < 1:
        module.fail_json(msg='no messages matching ts: %s' % ts)
    if len(data['messages']) > 1:
        module.fail_json(msg='more than 1 message matching ts: %s' % ts)
    return data['messages'][0]