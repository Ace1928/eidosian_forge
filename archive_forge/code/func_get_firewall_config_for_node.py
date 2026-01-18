from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import netapp_ipaddress
def get_firewall_config_for_node(self):
    """
        Get firewall configuration on the node
        :return: dict() with firewall config details
        """
    if self.parameters.get('logging') and self.parameters.get('node') is None:
        self.module.fail_json(msg="Error: Missing parameter 'node' to modify firewall logging")
    net_firewall_config_obj = netapp_utils.zapi.NaElement('net-firewall-config-get')
    net_firewall_config_obj.add_new_child('node-name', self.parameters['node'])
    try:
        result = self.server.invoke_successfully(net_firewall_config_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting Firewall Configuration: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('attributes'):
        firewall_info = result['attributes'].get_child_by_name('net-firewall-config-info')
        return {'enable': self.change_status_to_bool(firewall_info.get_child_content('is-enabled'), to_zapi=False), 'logging': self.change_status_to_bool(firewall_info.get_child_content('is-logging'), to_zapi=False)}
    return None