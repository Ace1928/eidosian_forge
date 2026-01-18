from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def add_uuid_subsets(self, get_ontap_subset_info):
    params = self.parameters.get('owning_resource')
    owning_resource_supported_subsets = ['storage/volumes/snapshots', 'protocols/nfs/export-policies/rules', 'protocols/vscan/on-access-policies', 'protocols/vscan/on-demand-policies', 'protocols/vscan/scanner-pools']
    if 'gather_subset' in self.parameters:
        if 'storage/volumes/snapshots' in self.parameters['gather_subset']:
            self.check_error_values('storage/volumes/snapshots', params, ['volume_name', 'svm_name'])
            volume_uuid = rest_owning_resource.get_volume_uuid(self.rest_api, self.parameters['owning_resource']['volume_name'], self.parameters['owning_resource']['svm_name'], self.module)
            if volume_uuid:
                get_ontap_subset_info['storage/volumes/snapshots'] = {'api_call': 'storage/volumes/%s/snapshots' % volume_uuid}
        if 'protocols/nfs/export-policies/rules' in self.parameters['gather_subset']:
            self.check_error_values('protocols/nfs/export-policies/rules', params, ['policy_name', 'svm_name', 'rule_index'])
            policy_id = rest_owning_resource.get_export_policy_id(self.rest_api, self.parameters['owning_resource']['policy_name'], self.parameters['owning_resource']['svm_name'], self.module)
            if policy_id:
                get_ontap_subset_info['protocols/nfs/export-policies/rules'] = {'api_call': 'protocols/nfs/export-policies/%s/rules/%s' % (policy_id, self.parameters['owning_resource']['rule_index'])}
        if 'protocols/vscan/on-access-policies' in self.parameters['gather_subset']:
            self.add_vserver_owning_resource('protocols/vscan/on-access-policies', params, 'protocols/vscan/%s/on-access-policies', get_ontap_subset_info)
        if 'protocols/vscan/on-demand-policies' in self.parameters['gather_subset']:
            self.add_vserver_owning_resource('protocols/vscan/on-demand-policies', params, 'protocols/vscan/%s/on-demand-policies', get_ontap_subset_info)
        if 'protocols/vscan/scanner-pools' in self.parameters['gather_subset']:
            self.add_vserver_owning_resource('protocols/vscan/scanner-pools', params, 'protocols/vscan/%s/scanner-pools', get_ontap_subset_info)
        owning_resource_warning = any((subset not in owning_resource_supported_subsets for subset in self.parameters['gather_subset']))
        if owning_resource_warning and params is not None:
            self.module.warn("Kindly refer to Ansible documentation to check the subsets that support option 'owning_resource'.")
    return get_ontap_subset_info