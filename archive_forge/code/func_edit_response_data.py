from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
import codecs
def edit_response_data(self, r, info, result, success_status):
    if r is not None:
        result['http_response_body'] = codecs.decode(r.read(), 'utf-8')
    elif 'body' in info:
        result['http_response_body'] = codecs.decode(info['body'], 'utf-8')
        del info['body']
    else:
        result['http_response_body'] = ''
    result['http_response_data'] = info
    result['nitro_errorcode'] = None
    result['nitro_message'] = None
    result['nitro_severity'] = None
    if result['http_response_body'] != '':
        try:
            data = self._module.from_json(result['http_response_body'])
        except ValueError:
            data = {}
        result['nitro_errorcode'] = data.get('errorcode')
        result['nitro_message'] = data.get('message')
        result['nitro_severity'] = data.get('severity')
    if result['nitro_errorcode'] is None:
        if result['http_response_data'].get('status') != success_status:
            result['nitro_errorcode'] = -1
            result['nitro_message'] = result['http_response_data'].get('msg', 'HTTP status %s' % result['http_response_data']['status'])
            result['nitro_severity'] = 'ERROR'
        else:
            result['nitro_errorcode'] = 0
            result['nitro_message'] = 'Success'
            result['nitro_severity'] = 'NONE'