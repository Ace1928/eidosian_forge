from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_fpolicy_ext_engine(self):
    """
        Check to see if the fPolicy external engine exists or not
        :return: dict of engine properties if exist, None if not
        """
    return_value = None
    if self.use_rest:
        fields = ['vserver', 'engine-name', 'primary-servers', 'port', 'secondary-servers', 'extern-engine-type', 'ssl-option', 'max-connection-retries', 'max-server-reqs', 'certificate-common-name', 'certificate-serial', 'certificate-ca', 'recv-buffer-size', 'send-buffer-size', 'is-resiliency-enabled', 'resiliency-directory-path']
        api = 'private/cli/vserver/fpolicy/policy/external-engine'
        query = {'fields': ','.join(fields), 'engine-name': self.parameters['name'], 'vserver': self.parameters['vserver']}
        message, error = self.rest_api.get(api, query)
        return_info, error = rrh.check_for_0_or_1_records(api, message, error)
        if return_info is None:
            return None
        return_value = message['records'][0]
        return return_value
    else:
        fpolicy_ext_engine_obj = netapp_utils.zapi.NaElement('fpolicy-policy-external-engine-get-iter')
        fpolicy_ext_engine_config = netapp_utils.zapi.NaElement('fpolicy-external-engine-info')
        fpolicy_ext_engine_config.add_new_child('engine-name', self.parameters['name'])
        fpolicy_ext_engine_config.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(fpolicy_ext_engine_config)
        fpolicy_ext_engine_obj.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(fpolicy_ext_engine_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error searching for fPolicy engine %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('attributes-list'):
            fpolicy_ext_engine_attributes = result['attributes-list']['fpolicy-external-engine-info']
            primary_servers = []
            primary_servers_elem = fpolicy_ext_engine_attributes.get_child_by_name('primary-servers')
            for primary_server in primary_servers_elem.get_children():
                primary_servers.append(primary_server.get_content())
            secondary_servers = []
            if fpolicy_ext_engine_attributes.get_child_by_name('secondary-servers'):
                secondary_servers_elem = fpolicy_ext_engine_attributes.get_child_by_name('secondary-servers')
                for secondary_server in secondary_servers_elem.get_children():
                    secondary_servers.append(secondary_server.get_content())
            return_value = {'vserver': fpolicy_ext_engine_attributes.get_child_content('vserver'), 'name': fpolicy_ext_engine_attributes.get_child_content('engine-name'), 'certificate_ca': fpolicy_ext_engine_attributes.get_child_content('certificate-ca'), 'certificate_common_name': fpolicy_ext_engine_attributes.get_child_content('certificate-common-name'), 'certificate_serial': fpolicy_ext_engine_attributes.get_child_content('certificate-serial'), 'extern_engine_type': fpolicy_ext_engine_attributes.get_child_content('extern-engine-type'), 'is_resiliency_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=fpolicy_ext_engine_attributes.get_child_content('is-resiliency-enabled')), 'max_connection_retries': self.na_helper.get_value_for_int(from_zapi=True, value=fpolicy_ext_engine_attributes.get_child_content('max-connection-retries')), 'max_server_reqs': self.na_helper.get_value_for_int(from_zapi=True, value=fpolicy_ext_engine_attributes.get_child_content('max-server-requests')), 'port': self.na_helper.get_value_for_int(from_zapi=True, value=fpolicy_ext_engine_attributes.get_child_content('port-number')), 'primary_servers': primary_servers, 'secondary_servers': secondary_servers, 'recv_buffer_size': self.na_helper.get_value_for_int(from_zapi=True, value=fpolicy_ext_engine_attributes.get_child_content('recv-buffer-size')), 'resiliency_directory_path': fpolicy_ext_engine_attributes.get_child_content('resiliency-directory-path'), 'send_buffer_size': self.na_helper.get_value_for_int(from_zapi=True, value=fpolicy_ext_engine_attributes.get_child_content('send-buffer-size')), 'ssl_option': fpolicy_ext_engine_attributes.get_child_content('ssl-option')}
    return return_value