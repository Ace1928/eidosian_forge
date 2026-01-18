from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_key_manager(self):
    """
        get key manager by ip address.
        :return: a dict of key manager
        """
    if self.use_rest:
        return self.get_key_manager_rest()
    key_manager_info = netapp_utils.zapi.NaElement('security-key-manager-get-iter')
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('key-manager-info', **{'key-manager-ip-address': self.parameters['ip_address']})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    key_manager_info.add_child_elem(query)
    try:
        result = self.cluster.invoke_successfully(key_manager_info, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching key manager: %s' % to_native(error), exception=traceback.format_exc())
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        key_manager = result.get_child_by_name('attributes-list').get_child_by_name('key-manager-info')
        return_value = {}
        if key_manager.get_child_by_name('key-manager-ip-address'):
            return_value['ip_address'] = key_manager.get_child_content('key-manager-ip-address')
        if key_manager.get_child_by_name('key-manager-server-status'):
            return_value['server_status'] = key_manager.get_child_content('key-manager-server-status')
        if key_manager.get_child_by_name('key-manager-tcp-port'):
            return_value['tcp_port'] = int(key_manager.get_child_content('key-manager-tcp-port'))
    return return_value