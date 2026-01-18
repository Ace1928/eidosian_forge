from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_public_key(self):
    api = 'security/authentication/publickeys'
    body = {'account.name': self.parameters['account'], 'public_key': self.parameters['public_key']}
    if self.parameters.get('vserver') is not None:
        body['owner.name'] = self.parameters['vserver']
    for attr in ('comment', 'index'):
        value = self.parameters.get(attr)
        if value is not None:
            body[attr] = value
    dummy, error = self.rest_api.post(api, body)
    if error:
        msg = 'Error in create_public_key: %s' % error
        self.module.fail_json(msg=msg)