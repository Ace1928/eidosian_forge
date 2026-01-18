from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_user_rest(self):
    api = 'security/accounts'
    query = {'name': self.parameters['name']}
    if self.parameters.get('vserver') is None:
        query['scope'] = 'cluster'
    else:
        query['owner.name'] = self.parameters['vserver']
    message, error = self.rest_api.get(api, query)
    if error:
        self.module.fail_json(msg='Error while fetching user info: %s' % error)
    if message['num_records'] == 1:
        return (message['records'][0]['owner']['uuid'], message['records'][0]['name'])
    if message['num_records'] > 1:
        self.module.fail_json(msg='Error while fetching user info, found multiple entries: %s' % repr(message))
    return None