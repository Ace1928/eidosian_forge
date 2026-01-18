from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cifs_local_group_rest(self):
    """
        Retrieves the local group of an SVM.
        """
    api = 'protocols/cifs/local-groups'
    query = {'name': self.parameters['group'], 'svm.name': self.parameters['vserver'], 'fields': 'svm.uuid,sid'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error on fetching cifs local-group: %s' % error)
    if record:
        self.svm_uuid = self.na_helper.safe_get(record, ['svm', 'uuid'])
        self.sid = self.na_helper.safe_get(record, ['sid'])
    if record is None:
        self.module.fail_json(msg='CIFS local group %s does not exist on vserver %s' % (self.parameters['group'], self.parameters['vserver']))