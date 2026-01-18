from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_ldap_client(self, client_config_name=None, vserver_name=None):
    """
        Checks if LDAP client config exists.

        :return:
            ldap client config object if found
            None if not found
        :rtype: object/None
        """
    client_config_info = netapp_utils.zapi.NaElement('ldap-client-get-iter')
    if client_config_name is None:
        client_config_name = self.parameters['name']
    if vserver_name is None:
        vserver_name = '*'
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('ldap-client', **{'ldap-client-config': client_config_name, 'vserver': vserver_name})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    client_config_info.add_child_elem(query)
    result = self.server.invoke_successfully(client_config_info, enable_tunneling=False)
    client_config_details = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        client_config_info = attributes_list.get_child_by_name('ldap-client')
        ldap_server_list = self.get_list_from_children(client_config_info, 'ldap-servers')
        preferred_ad_servers_list = self.get_list_from_children(client_config_info, 'preferred-ad-servers')
        client_config_details = {'name': client_config_info.get_child_content('ldap-client-config'), 'servers': ldap_server_list, 'ad_domain': client_config_info.get_child_content('ad-domain'), 'base_dn': client_config_info.get_child_content('base-dn'), 'base_scope': client_config_info.get_child_content('base-scope'), 'bind_as_cifs_server': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('bind-as-cifs-server')), 'bind_dn': client_config_info.get_child_content('bind-dn'), 'bind_password': client_config_info.get_child_content('bind-password'), 'min_bind_level': client_config_info.get_child_content('min-bind-level'), 'tcp_port': self.na_helper.get_value_for_int(from_zapi=True, value=client_config_info.get_child_content('tcp-port')), 'preferred_ad_servers': preferred_ad_servers_list, 'query_timeout': self.na_helper.get_value_for_int(from_zapi=True, value=client_config_info.get_child_content('query-timeout')), 'referral_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('referral-enabled')), 'schema': client_config_info.get_child_content('schema'), 'session_security': client_config_info.get_child_content('session-security'), 'use_start_tls': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('use-start-tls')), 'ldaps_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('ldaps-enabled'))}
    return client_config_details