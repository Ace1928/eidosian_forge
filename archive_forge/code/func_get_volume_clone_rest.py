from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_volume_clone_rest(self):
    api = 'storage/volumes'
    params = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver'], 'fields': 'clone.is_flexclone,uuid'}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error getting volume clone %s: %s' % (self.parameters['name'], to_native(error)))
    if record:
        return self.format_get_volume_clone_rest(record)
    return record