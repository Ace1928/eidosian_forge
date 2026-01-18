from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def sp_firmware_image_update(self):
    """
        Update current firmware image
        """
    firmware_update_info = netapp_utils.zapi.NaElement('service-processor-image-update')
    if self.parameters.get('package') is not None:
        firmware_update_info.add_new_child('package', self.parameters['package'])
    if self.parameters.get('clear_logs') is not None:
        firmware_update_info.add_new_child('clear-logs', str(self.parameters['clear_logs']))
    if self.parameters.get('install_baseline_image') is not None:
        firmware_update_info.add_new_child('install-baseline-image', str(self.parameters['install_baseline_image']))
    firmware_update_info.add_new_child('node', self.parameters['node'])
    firmware_update_info.add_new_child('update-type', self.parameters['update_type'])
    try:
        self.server.invoke_successfully(firmware_update_info, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) == '13001' and error.message.startswith('Service Processor update skipped'):
            return False
        self.module.fail_json(msg='Error updating firmware image for %s: %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
    return True