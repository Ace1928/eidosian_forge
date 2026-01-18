from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_s3_groups(self):
    self.get_svm_uuid()
    api = 'protocols/s3/services/%s/groups' % self.svm_uuid
    fields = ','.join(('name', 'comment', 'users.name', 'policies.name'))
    params = {'name': self.parameters['name'], 'fields': fields}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching S3 groups %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if record:
        self.group_id = record.get('id')
        return self.form_current(record)
    return record