from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def form_create_or_modify_body(self, params):
    body = {}
    if params.get('is_scan_mandatory') is not None:
        body['mandatory'] = params['is_scan_mandatory']
    if params.get('policy_status') is not None:
        body['enabled'] = params['policy_status']
    if params.get('max_file_size'):
        body['scope.max_file_size'] = params['max_file_size']
    if params.get('scan_files_with_no_ext') is not None:
        body['scope.scan_without_extension'] = params['scan_files_with_no_ext']
    if 'file_ext_to_exclude' in params:
        body['scope.exclude_extensions'] = params['file_ext_to_exclude']
    if 'file_ext_to_include' in params:
        body['scope.include_extensions'] = params['file_ext_to_include']
    if 'paths_to_exclude' in params:
        body['scope.exclude_paths'] = params['paths_to_exclude']
    if params.get('scan_readonly_volumes') is not None:
        body['scope.scan_readonly_volumes'] = params['scan_readonly_volumes']
    if params.get('only_execute_access') is not None:
        body['scope.only_execute_access'] = params['only_execute_access']
    return body