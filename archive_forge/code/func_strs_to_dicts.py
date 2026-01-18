from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def strs_to_dicts(self):
    """transform applications list of strs to a list of dicts if application_strs in use"""
    if 'application_dicts' in self.parameters:
        for application in self.parameters['application_dicts']:
            application['authentication_methods'].sort()
        self.parameters['applications'] = self.parameters['application_dicts']
        self.parameters['replace_existing_apps_and_methods'] = 'always'
    elif 'application_strs' in self.parameters:
        self.parameters['applications'] = [dict(application=application, authentication_methods=[self.parameters['authentication_method']], second_authentication_method=None) for application in self.parameters['application_strs']]
    else:
        self.parameters['applications'] = None