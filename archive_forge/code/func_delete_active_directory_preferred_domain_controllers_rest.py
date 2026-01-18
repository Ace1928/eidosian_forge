from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def delete_active_directory_preferred_domain_controllers_rest(self):
    """
        Removes the Active Directory preferred DC configuration from an SVM.
        """
    if self.rest_api.meets_rest_minimum_version(True, 9, 12, 0):
        api = 'protocols/active-directory/%s/preferred-domain-controllers/%s/%s' % (self.svm_uuid, self.parameters['fqdn'], self.parameters['server_ip'])
        record, error = rest_generic.delete_async(self.rest_api, api, None)
    else:
        api = 'private/cli/vserver/active-directory/preferred-dc/remove'
        body = {'vserver': self.parameters['vserver'], 'domain': self.parameters['fqdn'], 'preferred_dc': [self.parameters['server_ip']]}
        dummy, error = rest_generic.delete_async(self.rest_api, api, None, body)
    if error:
        self.module.fail_json(msg='Error on deleting Active Directory preferred DC configuration of an SVM: %s' % error)