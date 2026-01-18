from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_if_remote_volume_exists(self):
    """
        Validate existence of source volume
        :return: True if volume exists, False otherwise
        """
    self.set_source_cluster_connection()
    if self.src_use_rest:
        return self.check_if_remote_volume_exists_rest()
    volume_info = netapp_utils.zapi.NaElement('volume-get-iter')
    volume_attributes = netapp_utils.zapi.NaElement('volume-attributes')
    volume_id_attributes = netapp_utils.zapi.NaElement('volume-id-attributes')
    volume_id_attributes.add_new_child('name', self.parameters['source_volume'])
    volume_id_attributes.add_new_child('vserver-name', self.parameters['source_vserver'])
    volume_attributes.add_child_elem(volume_id_attributes)
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(volume_attributes)
    volume_info.add_child_elem(query)
    try:
        result = self.source_server.invoke_successfully(volume_info, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching source volume details %s: %s' % (self.parameters['source_volume'], to_native(error)), exception=traceback.format_exc())
    return bool(result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0)