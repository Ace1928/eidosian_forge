from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def firmware_image_get_iter(self):
    """
        Compose NaElement object to query current firmware version
        :return: NaElement object for firmware_image_get_iter with query
        """
    firmware_image_get = netapp_utils.zapi.NaElement('service-processor-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    firmware_image_info = netapp_utils.zapi.NaElement('service-processor-info')
    firmware_image_info.add_new_child('node', self.parameters['node'])
    query.add_child_elem(firmware_image_info)
    firmware_image_get.add_child_elem(query)
    return firmware_image_get