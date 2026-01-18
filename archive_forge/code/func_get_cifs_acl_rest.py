from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cifs_acl_rest(self, svm_uuid):
    """
        get details of the CIFS share acl with rest API.
        """
    if not self.use_rest:
        return self.get_cifs_acl()
    query = {'user_or_group': self.parameters.get('user_or_group')}
    ug_type = self.parameters.get('type')
    if ug_type:
        query['type'] = ug_type
    api = 'protocols/cifs/shares/%s/%s/acls' % (svm_uuid['uuid'], self.parameters.get('share_name'))
    fields = 'svm.uuid,user_or_group,type,permission'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error on fetching cifs shares acl: %s' % error)
    if record:
        return {'uuid': record['svm']['uuid'], 'share': record['share'], 'user_or_group': record['user_or_group'], 'type': record['type'], 'permission': record['permission']}
    return None