from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def quota_entry_set_rest(self):
    """
        quota_entry_set with rest API.
        for type: 'user' and 'group', quota_target is used.
        value for user, group and qtree should be passed as ''
        """
    if not self.use_rest:
        return self.quota_entry_set()
    body = {'svm.name': self.parameters.get('vserver'), 'volume.name': self.parameters.get('volume'), 'type': self.parameters.get('type'), 'qtree.name': self.parameters.get('qtree')}
    quota_target = self.parameters.get('quota_target')
    if self.parameters.get('type') == 'user':
        body['users.name'] = quota_target.split(',')
    elif self.parameters.get('type') == 'group':
        body['group.name'] = quota_target
    if self.parameters.get('type') == 'tree':
        body['qtree.name'] = quota_target
    if 'file_limit' in self.parameters:
        body['files.hard_limit'] = self.parameters.get('file_limit')
    if 'soft_file_limit' in self.parameters:
        body['files.soft_limit'] = self.parameters.get('soft_file_limit')
    if 'disk_limit' in self.parameters:
        body['space.hard_limit'] = self.parameters.get('disk_limit')
    if 'soft_disk_limit' in self.parameters:
        body['space.soft_limit'] = self.parameters.get('soft_disk_limit')
    if 'perform_user_mapping' in self.parameters:
        body['user_mapping'] = self.parameters.get('perform_user_mapping')
    query = {'return_records': 'true'}
    api = 'storage/quota/rules'
    response, error = rest_generic.post_async(self.rest_api, api, body, query)
    if error:
        if 'job reported error:' in error and "entry doesn't exist" in error:
            self.module.warn('Ignoring job status, assuming success.')
        elif '5308568' in error:
            self.form_warn_msg_rest('create', '5308568')
        elif '5308571' in error:
            self.form_warn_msg_rest('create', '5308571')
        else:
            self.module.fail_json(msg='Error on creating quotas rule: %s' % error)
        self.volume_uuid = self.get_quota_status_or_volume_id_rest(get_volume=True)
    if not self.volume_uuid and response:
        record, error = rrh.check_for_0_or_1_records(api, response, error, query)
        if not error and record and (not record['volume']['uuid']):
            error = 'volume uuid key not present in %s:' % record
        if error:
            self.module.fail_json(msg='Error on getting volume uuid: %s' % error)
        if record:
            self.volume_uuid = record['volume']['uuid']