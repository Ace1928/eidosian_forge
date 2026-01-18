from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def format_on_demand_task(self, record):
    return {'task_name': record['name'], 'file_ext_to_exclude': self.na_helper.safe_get(record, ['scope', 'exclude_extensions']), 'file_ext_to_include': self.na_helper.safe_get(record, ['scope', 'include_extensions']), 'max_file_size': self.na_helper.safe_get(record, ['scope', 'max_file_size']), 'paths_to_exclude': self.na_helper.safe_get(record, ['scope', 'exclude_paths']), 'report_directory': self.na_helper.safe_get(record, ['log_path']), 'scan_files_with_no_ext': self.na_helper.safe_get(record, ['scope', 'scan_without_extension']), 'scan_paths': self.na_helper.safe_get(record, ['scan_paths']), 'schedule': self.na_helper.safe_get(record, ['schedule', 'name'])}