from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def download_software(self):
    if self.use_rest:
        return self.download_software_rest()
    package_exists = self.cluster_image_package_download()
    if package_exists is False:
        cluster_download_progress = self.cluster_image_package_download_progress()
        while cluster_download_progress is None or cluster_download_progress.get('progress_status') == 'async_pkg_get_phase_running':
            time.sleep(10)
            cluster_download_progress = self.cluster_image_package_download_progress()
        if cluster_download_progress.get('progress_status') != 'async_pkg_get_phase_complete':
            self.module.fail_json(msg='Error downloading package: %s - installed versions: %s' % (cluster_download_progress['failure_reason'], self.versions))