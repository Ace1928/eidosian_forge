from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
def checkID(module, params):
    data = urlencode(params)
    full_uri = API_BASE + API_ACTIONS['status'] + data
    req, info = fetch_url(module, full_uri)
    result = to_text(req.read())
    jsonresult = json.loads(result)
    req.close()
    return jsonresult