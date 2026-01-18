from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_volume_clone_rest(self):
    api = 'storage/volumes'
    body = {'name': self.parameters['name'], 'clone.parent_volume.name': self.parameters['parent_volume'], 'clone.is_flexclone': True, 'svm.name': self.parameters['vserver']}
    if self.parameters.get('qos_policy_group_name'):
        body['qos.policy.name'] = self.parameters['qos_policy_group_name']
    if self.parameters.get('parent_snapshot'):
        body['clone.parent_snapshot.name'] = self.parameters['parent_snapshot']
    if self.parameters.get('parent_vserver'):
        body['clone.parent_svm.name'] = self.parameters['parent_vserver']
    if self.parameters.get('volume_type'):
        body['type'] = self.parameters['volume_type']
    if self.parameters.get('junction_path'):
        body['nas.path'] = self.parameters['junction_path']
    if self.parameters.get('uid'):
        body['nas.uid'] = self.parameters['uid']
    if self.parameters.get('gid'):
        body['nas.gid'] = self.parameters['gid']
    query = {'return_records': 'true'}
    response, error = rest_generic.post_async(self.rest_api, api, body, query, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating volume clone %s: %s' % (self.parameters['name'], to_native(error)))
    if response:
        record, error = rrh.check_for_0_or_1_records(api, response, error, query)
        if not error and record and ('uuid' not in record):
            error = 'uuid key not present in %s:' % record
        if error:
            self.module.fail_json(msg='Error: failed to parse create clone response: %s' % error)
        if record:
            self.uuid = record['uuid']