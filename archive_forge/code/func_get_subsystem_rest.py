from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_subsystem_rest(self):
    api = 'protocols/nvme/subsystems'
    params = {'svm.name': self.parameters['vserver'], 'name': self.parameters['subsystem']}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        if self.na_helper.ignore_missing_vserver_on_delete(error):
            return None
        self.module.fail_json(msg='Error fetching subsystem info for vserver: %s, %s' % (self.parameters['vserver'], to_native(error)))
    if record:
        self.subsystem_uuid = record['uuid']
        return record
    return None