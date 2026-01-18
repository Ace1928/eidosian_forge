from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_net_port(self, port, modify):
    """
        Modify a port

        :param port: Name of the port
        :param modify: dict with attributes to be modified
        :return: None
        """
    if self.use_rest:
        return self.modify_net_port_rest(port, modify)

    def get_zapi_key_and_value(key, value):
        zapi_key = self.na_helper.zapi_string_keys.get(key)
        if zapi_key is not None:
            return (zapi_key, value)
        zapi_key = self.na_helper.zapi_bool_keys.get(key)
        if zapi_key is not None:
            return (zapi_key, self.na_helper.get_value_for_bool(from_zapi=False, value=value))
        zapi_key = self.na_helper.zapi_int_keys.get(key)
        if zapi_key is not None:
            return (zapi_key, self.na_helper.get_value_for_int(from_zapi=False, value=value))
        raise KeyError(key)
    port_modify = netapp_utils.zapi.NaElement('net-port-modify')
    port_attributes = {'node': self.parameters['node'], 'port': port}
    for key, value in modify.items():
        zapi_key, value = get_zapi_key_and_value(key, value)
        port_attributes[zapi_key] = value
    port_modify.translate_struct(port_attributes)
    try:
        self.server.invoke_successfully(port_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying net ports for %s: %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())