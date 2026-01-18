from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def download_sp_image_progress(self):
    progress = netapp_utils.zapi.NaElement('system-image-update-progress-get')
    progress.add_new_child('node', self.parameters['node'])
    progress_info = {}
    try:
        result = self.server.invoke_successfully(progress, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching system image package download progress: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('phase'):
        progress_info['phase'] = result.get_child_content('phase')
    else:
        progress_info['phase'] = None
    if result.get_child_by_name('exit-message'):
        progress_info['exit_message'] = result.get_child_content('exit-message')
    else:
        progress_info['exit_message'] = None
    if result.get_child_by_name('exit-status'):
        progress_info['exit_status'] = result.get_child_content('exit-status')
    else:
        progress_info['exit_status'] = None
    if result.get_child_by_name('last-message'):
        progress_info['last_message'] = result.get_child_content('last-message')
    else:
        progress_info['last_message'] = None
    if result.get_child_by_name('run-status'):
        progress_info['run_status'] = result.get_child_content('run-status')
    else:
        progress_info['run_status'] = None
    return progress_info