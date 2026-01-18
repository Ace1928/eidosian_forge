from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_active_directory_preferred_domain_controllers_rest(self):
    """
        Retrieves the Active Directory preferred DC configuration of an SVM.
        """
    if self.rest_api.meets_rest_minimum_version(True, 9, 12, 0):
        api = 'protocols/active-directory/%s/preferred-domain-controllers' % self.svm_uuid
        query = {'svm.name': self.parameters['vserver'], 'fqdn': self.parameters['fqdn'], 'server_ip': self.parameters['server_ip'], 'fields': 'server_ip,fqdn'}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error on fetching Active Directory preferred DC configuration of an SVM: %s' % error)
        if record:
            return record
    else:
        api = 'private/cli/vserver/active-directory/preferred-dc'
        query = {'vserver': self.parameters['vserver'], 'domain': self.parameters['fqdn'], 'preferred_dc': self.parameters['server_ip'], 'fields': 'domain,preferred-dc'}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error on fetching Active Directory preferred DC configuration of an SVM using cli: %s' % error)
        if record:
            return {'server_ip': self.na_helper.safe_get(record, ['preferred_dc']), 'fqdn': self.na_helper.safe_get(record, ['domain'])}
    return None