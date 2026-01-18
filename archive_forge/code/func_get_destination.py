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
def get_destination(self):
    """
        get the destination info
        # Note: REST module to get_destination is not required as it's used in only ZAPI.
        """
    result = None
    get_dest_iter = netapp_utils.zapi.NaElement('snapmirror-get-destination-iter')
    query = netapp_utils.zapi.NaElement('query')
    snapmirror_dest_info = netapp_utils.zapi.NaElement('snapmirror-destination-info')
    snapmirror_dest_info.add_new_child('destination-location', self.parameters['destination_path'])
    query.add_child_elem(snapmirror_dest_info)
    get_dest_iter.add_child_elem(query)
    try:
        result = self.source_server.invoke_successfully(get_dest_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching snapmirror destinations info: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        return True
    return None