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
def aggr_get_iter(self, name):
    """
        Return aggr-get-iter query results
        :param name: Name of the aggregate
        :return: NaElement if aggregate found, None otherwise
        """
    aggr_get_iter = netapp_utils.zapi.NaElement('aggr-get-iter')
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('aggr-attributes', **{'aggregate-name': name})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    aggr_get_iter.add_child_elem(query)
    result = None
    try:
        result = self.server.invoke_successfully(aggr_get_iter, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) != '13040':
            self.module.fail_json(msg='Error getting aggregate: %s' % to_native(error), exception=traceback.format_exc())
    return result