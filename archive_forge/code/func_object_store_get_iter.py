from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def object_store_get_iter(self, name):
    """
        Return aggr-object-store-get query results
        :return: NaElement if object-store for given aggregate found, None otherwise
        """
    object_store_get_iter = netapp_utils.zapi.NaElement('aggr-object-store-get-iter')
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('object-store-information', **{'object-store-name': self.parameters.get('object_store_name'), 'aggregate': name})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    object_store_get_iter.add_child_elem(query)
    result = None
    try:
        result = self.server.invoke_successfully(object_store_get_iter, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting object store: %s' % to_native(error), exception=traceback.format_exc())
    return result