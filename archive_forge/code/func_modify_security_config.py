from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_security_config(self, modify):
    """
        Modifies the security configuration.
        """
    if self.use_rest:
        return self.modify_security_config_rest(modify)
    security_config_obj = netapp_utils.zapi.NaElement('security-config-modify')
    security_config_obj.add_new_child('interface', self.parameters['name'])
    if 'is_fips_enabled' in self.parameters:
        self.parameters['is_fips_enabled'] = self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_fips_enabled'])
        security_config_obj.add_new_child('is-fips-enabled', self.parameters['is_fips_enabled'])
    if 'supported_ciphers' in self.parameters:
        security_config_obj.add_new_child('supported-ciphers', self.parameters['supported_ciphers'])
    if 'supported_protocols' in self.parameters:
        supported_protocol_obj = netapp_utils.zapi.NaElement('supported-protocols')
        for protocol in self.parameters['supported_protocols']:
            supported_protocol_obj.add_new_child('string', protocol)
        security_config_obj.add_child_elem(supported_protocol_obj)
    try:
        self.server.invoke_successfully(security_config_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying security config for interface %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())