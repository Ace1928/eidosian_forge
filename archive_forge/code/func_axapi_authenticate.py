from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url
def axapi_authenticate(module, base_url, username, password):
    url = '%s&method=authenticate&username=%s&password=%s' % (base_url, username, password)
    result = axapi_call(module, url)
    if axapi_failure(result):
        return module.fail_json(msg=result['response']['err']['msg'])
    sessid = result['session_id']
    return base_url + '&session_id=' + sessid