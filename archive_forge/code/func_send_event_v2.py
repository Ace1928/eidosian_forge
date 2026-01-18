from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, urlunparse
from datetime import datetime
def send_event_v2(module, service_key, event_type, payload, link, incident_key=None, client=None, client_url=None):
    url = 'https://events.pagerduty.com/v2/enqueue'
    headers = {'Content-type': 'application/json'}
    data = {'routing_key': service_key, 'event_action': event_type, 'payload': payload, 'client': client, 'client_url': client_url}
    if link:
        data['links'] = [link]
    if incident_key:
        data['dedup_key'] = incident_key
    if event_type != 'trigger':
        data.pop('payload')
    response, info = fetch_url(module, url, method='post', headers=headers, data=json.dumps(data))
    if info['status'] != 202:
        module.fail_json(msg='failed to %s. Reason: %s' % (event_type, info['msg']))
    json_out = json.loads(response.read())
    return (json_out, True)