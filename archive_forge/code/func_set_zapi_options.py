from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def set_zapi_options(self, options):
    if self.parameters.get('file_limit'):
        options['file-limit'] = self.parameters['file_limit']
    if self.parameters.get('disk_limit'):
        options['disk-limit'] = self.parameters['disk_limit']
    if self.parameters.get('perform_user_mapping') is not None:
        options['perform-user-mapping'] = str(self.parameters['perform_user_mapping'])
    if self.parameters.get('soft_file_limit'):
        options['soft-file-limit'] = self.parameters['soft_file_limit']
    if self.parameters.get('soft_disk_limit'):
        options['soft-disk-limit'] = self.parameters['soft_disk_limit']
    if self.parameters.get('threshold'):
        options['threshold'] = self.parameters['threshold']