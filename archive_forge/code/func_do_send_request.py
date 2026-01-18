from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
def do_send_request(module, url, params, key):
    data = json.dumps(params)
    headers = {'Content-Type': 'application/json', 'x-stackdriver-apikey': key}
    response, info = fetch_url(module, url, headers=headers, data=data, method='POST')
    if info['status'] != 200:
        module.fail_json(msg='Unable to send msg: %s' % info['msg'])