from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
import codecs
def handle_get_return_object(self, result):
    result['nitro_object'] = []
    if result['nitro_errorcode'] == 0:
        if result['http_response_body'] != '':
            data = self._module.from_json(result['http_response_body'])
            if self._module.params['resource'] in data:
                result['nitro_object'] = data[self._module.params['resource']]
    else:
        del result['nitro_object']