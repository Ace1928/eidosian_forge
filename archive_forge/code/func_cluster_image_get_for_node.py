from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_get_for_node(self, node_name):
    """
        Get current cluster image info for given node
        """
    cluster_image_get = netapp_utils.zapi.NaElement('cluster-image-get')
    cluster_image_get.add_new_child('node-id', node_name)
    try:
        result = self.server.invoke_successfully(cluster_image_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching cluster image details for %s: %s' % (node_name, to_native(error)), exception=traceback.format_exc())
    image_info = self.na_helper.safe_get(result, ['attributes', 'cluster-image-info'])
    if image_info:
        return (image_info.get_child_content('node-id'), image_info.get_child_content('current-version'))
    return (None, None)