from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_lun_map_reporting_nodes(self):
    """
        Return list of reporting nodes from the LUN map

        :return: list of reporting nodes
        :rtype: list
        """
    if self.use_rest:
        return self.get_lun_map_reporting_nodes_rest()
    query_details = netapp_utils.zapi.NaElement('lun-map-info')
    query_details.add_new_child('path', self.parameters['path'])
    query_details.add_new_child('initiator-group', self.parameters['initiator_group_name'])
    query_details.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    lun_query = netapp_utils.zapi.NaElement('lun-map-get-iter')
    lun_query.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(lun_query, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting LUN map for %s: %s' % (self.parameters['initiator_group_name'], to_native(error)), exception=traceback.format_exc())
    try:
        num_records = int(result.get_child_content('num-records'))
    except TypeError:
        self.module.fail_json(msg='Error: unexpected ZAPI response for lun-map-get-iter: %s' % result.to_string())
    if num_records == 0:
        return None
    alist = result.get_child_by_name('attributes-list')
    info = alist.get_child_by_name('lun-map-info')
    reporting_nodes = info.get_child_by_name('reporting-nodes')
    node_list = []
    if reporting_nodes:
        for node in reporting_nodes.get_children():
            node_list.append(node.get_content())
    return node_list