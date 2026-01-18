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
def get_quotas_rest(self):
    """
        Retrieves quotas with rest API.
        If type is user then it returns all possible combinations of user name records.
        Report api is used to fetch file and disk limit info
        """
    if not self.use_rest:
        return self.get_quotas()
    query = {'svm.name': self.parameters.get('vserver'), 'volume.name': self.parameters.get('volume'), 'type': self.parameters.get('type'), 'fields': 'svm.uuid,svm.name,space.hard_limit,files.hard_limit,user_mapping,qtree.name,type,space.soft_limit,files.soft_limit,volume.uuid,users.name,group.name,'}
    if self.parameters['qtree']:
        query['qtree.name'] = self.parameters['qtree']
    if self.parameters.get('quota_target'):
        type = self.parameters['type']
        field_name = 'users.name' if type == 'user' else 'group.name' if type == 'group' else 'qtree.name'
        query[field_name] = self.parameters['quota_target']
    api = 'storage/quota/rules'
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error on getting quota rule info: %s' % error)
    if records:
        record = None
        for item in records:
            desired_qtree = self.parameters['qtree'] if self.parameters.get('qtree') else None
            current_qtree = self.na_helper.safe_get(item, ['qtree', 'name'])
            type = self.parameters.get('type')
            if type in ['user', 'group']:
                if desired_qtree != current_qtree:
                    continue
                if type == 'user':
                    desired_users = self.parameters['quota_target'].split(',')
                    current_users = [user['name'] for user in item['users']]
                    if set(current_users) == set(desired_users):
                        record = item
                        break
                elif item['group']['name'] == self.parameters['quota_target']:
                    record = item
                    break
            elif type == 'tree' and current_qtree == self.parameters['quota_target']:
                record = item
                break
        if record:
            self.volume_uuid = record['volume']['uuid']
            self.quota_uuid = record['uuid']
            current = {'soft_file_limit': self.na_helper.safe_get(record, ['files', 'soft_limit']), 'disk_limit': self.na_helper.safe_get(record, ['space', 'hard_limit']), 'soft_disk_limit': self.na_helper.safe_get(record, ['space', 'soft_limit']), 'file_limit': self.na_helper.safe_get(record, ['files', 'hard_limit']), 'perform_user_mapping': self.na_helper.safe_get(record, ['user_mapping'])}
            current['soft_file_limit'] = '-' if current['soft_file_limit'] is None else str(current['soft_file_limit'])
            current['disk_limit'] = '-' if current['disk_limit'] is None else str(current['disk_limit'])
            current['soft_disk_limit'] = '-' if current['soft_disk_limit'] is None else str(current['soft_disk_limit'])
            current['file_limit'] = '-' if current['file_limit'] is None else str(current['file_limit'])
            return current
    return None