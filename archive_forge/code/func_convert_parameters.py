from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def convert_parameters(self):
    if self.parameters.get('privileges') is not None:
        return
    if not self.parameters.get('command_directory_name'):
        self.module.fail_json(msg='Error: either path or command_directory_name is required in REST.')
    self.parameters['privileges'] = []
    temp_dict = {'path': self.parameters['command_directory_name'], 'access': self.parameters['access_level']}
    self.parameters.pop('command_directory_name')
    self.parameters.pop('access_level')
    if self.parameters.get('query'):
        temp_dict['query'] = self.parameters['query']
        self.parameters.pop('query')
    self.parameters['privileges'] = [temp_dict]