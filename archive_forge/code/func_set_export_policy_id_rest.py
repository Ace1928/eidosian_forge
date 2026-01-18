from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def set_export_policy_id_rest(self):
    if self.policy_id is not None:
        return
    options = {'fields': 'name,id', 'svm.name': self.parameters['vserver'], 'name': self.parameters['name']}
    api = 'protocols/nfs/export-policies'
    record, error = rest_generic.get_one_record(self.rest_api, api, options)
    if error:
        self.module.fail_json(msg='Error on fetching export policy: %s' % error)
    if record:
        self.policy_id = record['id']