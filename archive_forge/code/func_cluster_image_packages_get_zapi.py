from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_packages_get_zapi(self):
    versions = []
    packages_obj = netapp_utils.zapi.NaElement('cluster-image-package-local-get-iter')
    try:
        result = self.server.invoke_successfully(packages_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting list of local packages: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        packages_info = result.get_child_by_name('attributes-list')
        versions = [packages_details.get_child_content('package-version') for packages_details in packages_info.get_children()]
    return (versions, None)