from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_ldap(self, client_config_name=None):
    """
        Checks if LDAP config exists.

        :return:
            ldap config object if found
            None if not found
        :rtype: object/None
        """
    config_info = netapp_utils.zapi.NaElement('ldap-config-get-iter')
    if client_config_name is None:
        client_config_name = self.parameters['name']
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('ldap-config', **{'client-config': client_config_name})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    config_info.add_child_elem(query)
    result = self.server.invoke_successfully(config_info, enable_tunneling=True)
    config_details = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        config_info = attributes_list.get_child_by_name('ldap-config')
        config_details = {'client_config': config_info.get_child_content('client-config'), 'skip_config_validation': config_info.get_child_content('skip-config-validation'), 'vserver': config_info.get_child_content('vserver')}
    return config_details