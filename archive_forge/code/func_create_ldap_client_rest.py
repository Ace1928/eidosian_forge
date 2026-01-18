from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_ldap_client_rest(self):
    """
        create ldap client config with rest API.
        """
    if not self.use_rest:
        return self.create_ldap_client()
    body = self.create_ldap_client_body_rest()
    body['svm.name'] = self.parameters['vserver']
    api = 'name-services/ldap'
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error on creating ldap client: %s' % error)