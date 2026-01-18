from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_cifs_local_user(self):
    self.get_svm_uuid()
    api = 'protocols/cifs/local-users'
    fields = 'account_disabled,description,full_name,name,sid'
    params = {'svm.uuid': self.svm_uuid, 'name': self.parameters['name'], 'fields': fields}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching cifs/local-user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if record:
        return self.format_record(record)
    return None