from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_net_port(self, port):
    """
        Return details about the net port
        :param: port: Name of the port
        :return: Dictionary with current state of the port. None if not found.
        :rtype: dict
        """
    if self.use_rest:
        return self.get_net_port_rest(port)
    net_port_get = netapp_utils.zapi.NaElement('net-port-get-iter')
    attributes = {'query': {'net-port-info': {'node': self.parameters['node'], 'port': port}}}
    net_port_get.translate_struct(attributes)
    try:
        result = self.server.invoke_successfully(net_port_get, True)
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            port_info = result['attributes-list']['net-port-info']
            port_details = dict()
        else:
            return None
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting net ports for %s: %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
    for item_key, zapi_key in self.na_helper.zapi_bool_keys.items():
        port_details[item_key] = self.na_helper.get_value_for_bool(from_zapi=True, value=port_info.get_child_content(zapi_key))
    for item_key, zapi_key in self.na_helper.zapi_int_keys.items():
        port_details[item_key] = self.na_helper.get_value_for_int(from_zapi=True, value=port_info.get_child_content(zapi_key))
    for item_key, zapi_key in self.na_helper.zapi_string_keys.items():
        port_details[item_key] = port_info.get_child_content(zapi_key)
    return port_details