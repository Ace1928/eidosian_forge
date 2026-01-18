from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
@staticmethod
def cluster_image_get_iter():
    """
        Compose NaElement object to query current version
        :return: NaElement object for cluster-image-get-iter with query
        """
    cluster_image_get = netapp_utils.zapi.NaElement('cluster-image-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    cluster_image_info = netapp_utils.zapi.NaElement('cluster-image-info')
    query.add_child_elem(cluster_image_info)
    cluster_image_get.add_child_elem(query)
    return cluster_image_get