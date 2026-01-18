from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_demand_task_rest(self):
    api = 'protocols/vscan/%s/on-demand-policies' % self.svm_uuid
    query = {'name': self.parameters['task_name'], 'fields': 'scope.exclude_extensions,scope.include_extensions,scope.max_file_size,scope.exclude_paths,log_path,scope.scan_without_extension,scan_paths,schedule.name,name'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error fetching on demand task %s: %s' % (self.parameters['task_name'], to_native(error)))
    if record:
        return self.format_on_demand_task(record)
    return None