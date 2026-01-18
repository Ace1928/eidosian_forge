from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_body(self, params):
    body = self.create_body_or_query_common(params)
    if params.get('protocol'):
        body['protocols'] = self.parameters['protocol']
    if params.get('super_user_security'):
        body['superuser'] = self.parameters['super_user_security']
    if params.get('client_match'):
        body['clients'] = self.client_match_format(self.parameters['client_match'])
    if params.get('ro_rule'):
        body['ro_rule'] = self.parameters['ro_rule']
    if params.get('rw_rule'):
        body['rw_rule'] = self.parameters['rw_rule']
    return body