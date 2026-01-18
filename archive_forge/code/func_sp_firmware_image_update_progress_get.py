from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def sp_firmware_image_update_progress_get(self, node_name):
    """
        Get current firmware image update progress info
        :return: Dictionary of firmware image update progress if query successful, else return None
        """
    firmware_update_progress_get = netapp_utils.zapi.NaElement('service-processor-image-update-progress-get')
    firmware_update_progress_get.add_new_child('node', self.parameters['node'])
    firmware_update_progress_info = {}
    try:
        result = self.server.invoke_successfully(firmware_update_progress_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching firmware image upgrade progress details: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('attributes').get_child_by_name('service-processor-image-update-progress-info'):
        update_progress_info = result.get_child_by_name('attributes').get_child_by_name('service-processor-image-update-progress-info')
        firmware_update_progress_info['is-in-progress'] = update_progress_info.get_child_content('is-in-progress')
        firmware_update_progress_info['node'] = update_progress_info.get_child_content('node')
    return firmware_update_progress_info