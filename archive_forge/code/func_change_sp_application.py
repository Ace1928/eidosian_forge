from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def change_sp_application(self, current_apps):
    """Adjust requested app name to match ONTAP convention"""
    if not self.parameters['applications']:
        return
    app_list = [app['application'] for app in current_apps]
    for application in self.parameters['applications']:
        if application['application'] == 'service_processor' and 'service-processor' in app_list:
            application['application'] = 'service-processor'
        elif application['application'] == 'service-processor' and 'service_processor' in app_list:
            application['application'] = 'service_processor'