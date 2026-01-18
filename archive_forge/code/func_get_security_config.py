from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_security_config(self):
    """
            Get the current security configuration
        """
    if self.use_rest:
        return self.get_security_config_rest()
    return_value = None
    security_config_get_iter = netapp_utils.zapi.NaElement('security-config-get')
    security_config_info = netapp_utils.zapi.NaElement('desired-attributes')
    if 'is_fips_enabled' in self.parameters:
        security_config_info.add_new_child('is-fips-enabled', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_fips_enabled']))
    if 'supported_ciphers' in self.parameters:
        security_config_info.add_new_child('supported-ciphers', self.parameters['supported_ciphers'])
    if 'supported_protocols' in self.parameters:
        security_config_info.add_new_child('supported-protocols', ','.join(self.parameters['supported_protocols']))
    security_config_get_iter.add_child_elem(security_config_info)
    security_config_get_iter.add_new_child('interface', self.parameters['name'])
    try:
        result = self.server.invoke_successfully(security_config_get_iter, True)
        security_supported_protocols = []
        if result.get_child_by_name('attributes'):
            attributes = result.get_child_by_name('attributes')
            security_config_attributes = attributes.get_child_by_name('security-config-info')
            supported_protocols = security_config_attributes.get_child_by_name('supported-protocols')
            for supported_protocol in supported_protocols.get_children():
                security_supported_protocols.append(supported_protocol.get_content())
            return_value = {'name': security_config_attributes['interface'], 'is_fips_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=security_config_attributes['is-fips-enabled']), 'supported_ciphers': security_config_attributes['supported-ciphers'], 'supported_protocols': security_supported_protocols}
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting security config for interface %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return return_value