from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_home_node_for_cluster(self):
    """ get the first node name from this cluster """
    if self.use_rest:
        if not self.home_node:
            nodes = self.get_cluster_node_names_rest()
            if nodes:
                self.home_node = nodes[0]
        return self.home_node
    get_node = netapp_utils.zapi.NaElement('cluster-node-get-iter')
    attributes = {'query': {'cluster-node-info': {}}}
    get_node.translate_struct(attributes)
    try:
        result = self.server.invoke_successfully(get_node, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as exc:
        if str(exc.code) == '13003' or exc.message == 'ZAPI is not enabled in pre-cluster mode.':
            return None
        self.module.fail_json(msg='Error fetching node for interface %s: %s' % (self.parameters['interface_name'], to_native(exc)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes = result.get_child_by_name('attributes-list')
        return attributes.get_child_by_name('cluster-node-info').get_child_content('node-name')
    return None