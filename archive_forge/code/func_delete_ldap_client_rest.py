from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def delete_ldap_client_rest(self, current):
    """
        delete ldap client config with rest API.
        """
    if not self.use_rest:
        return self.delete_ldap_client()
    api = 'name-services/ldap'
    dummy, error = rest_generic.delete_async(self.rest_api, api, current['svm']['uuid'], body=None)
    if error is not None:
        self.module.fail_json(msg='Error on deleting ldap client rest: %s' % error)