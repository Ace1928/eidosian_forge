from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_unix_user_rest(self):
    """
        Retrieves UNIX user information for the specified user and SVM with rest API.
        """
    if not self.use_rest:
        return self.get_unix_user()
    query = {'svm.name': self.parameters.get('vserver'), 'name': self.parameters.get('name')}
    api = 'name-services/unix-users'
    fields = 'svm.uuid,id,primary_gid,name,full_name'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error on getting unix-user info: %s' % error)
    if record:
        return {'svm': {'uuid': self.na_helper.safe_get(record, ['svm', 'uuid'])}, 'name': self.na_helper.safe_get(record, ['name']), 'full_name': self.na_helper.safe_get(record, ['full_name']), 'id': self.na_helper.safe_get(record, ['id']), 'primary_gid': self.na_helper.safe_get(record, ['primary_gid'])}
    return None