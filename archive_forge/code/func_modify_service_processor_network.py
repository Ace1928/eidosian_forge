from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import time
def modify_service_processor_network(self, params=None):
    """
        Modify a service processor network.
        :param params: A dict of modified options.
        When dhcp is not set to v4, ip_address, netmask, and gateway_ip_address must be specified even if remains the same.
        """
    if self.use_rest:
        return self.modify_service_processor_network_rest(params)
    sp_modify = netapp_utils.zapi.NaElement('service-processor-network-modify')
    sp_attributes = dict()
    for item_key in self.parameters:
        if item_key in self.na_helper.zapi_string_keys:
            zapi_key = self.na_helper.zapi_string_keys.get(item_key)
            sp_attributes[zapi_key] = self.parameters[item_key]
        elif item_key in self.na_helper.zapi_bool_keys:
            zapi_key = self.na_helper.zapi_bool_keys.get(item_key)
            sp_attributes[zapi_key] = self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters[item_key])
        elif item_key in self.na_helper.zapi_int_keys:
            zapi_key = self.na_helper.zapi_int_keys.get(item_key)
            sp_attributes[zapi_key] = self.na_helper.get_value_for_int(from_zapi=False, value=self.parameters[item_key])
    sp_modify.translate_struct(sp_attributes)
    try:
        self.server.invoke_successfully(sp_modify, enable_tunneling=True)
        if self.parameters.get('wait_for_completion'):
            retries = 25
            status_key = 'not_setup' if params.get('is_enabled') else 'in_progress'
            while self.get_sp_network_status() == status_key and retries > 0:
                time.sleep(15)
                retries -= 1
            time.sleep(10)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying service processor network: %s' % to_native(error), exception=traceback.format_exc())