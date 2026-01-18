from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_package_download(self):
    """
        Get current cluster image package download
        :return: True if package already exists, else return False
        """
    cluster_image_package_download_info = netapp_utils.zapi.NaElement('cluster-image-package-download')
    cluster_image_package_download_info.add_new_child('package-url', self.parameters['package_url'])
    try:
        self.server.invoke_successfully(cluster_image_package_download_info, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) == '18408':
            return self.check_for_existing_package(error)
        else:
            self.module.fail_json(msg='Error downloading cluster image package for %s: %s' % (self.parameters['package_url'], to_native(error)), exception=traceback.format_exc())
    return False