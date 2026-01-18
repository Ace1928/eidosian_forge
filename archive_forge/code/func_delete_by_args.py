from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
import codecs
def delete_by_args(self):
    if self._module.params['resource'] is None:
        self.fail_module(msg='NITRO resource is undefined.')
    if self._module.params['args'] is None:
        self.fail_module(msg='NITRO args is undefined.')
    url = '%s://%s/nitro/v1/config/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'])
    args_dict = self._module.params['args']
    args = ','.join(['%s:%s' % (k, args_dict[k]) for k in args_dict])
    args = 'args=' + args
    url = '?'.join([url, args])
    r, info = fetch_url(self._module, url=url, headers=self._headers, method='DELETE')
    result = {}
    self.edit_response_data(r, info, result, success_status=200)
    if result['nitro_errorcode'] == 0:
        self._module_result['changed'] = True
    else:
        self._module_result['changed'] = False
    return result