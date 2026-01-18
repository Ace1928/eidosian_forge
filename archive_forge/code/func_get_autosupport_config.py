from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_autosupport_config(self):
    """
        get current autosupport details
        :return: dict()
        """
    asup_info = {}
    if self.use_rest:
        api = 'private/cli/system/node/autosupport'
        query = {'node': self.parameters['node_name'], 'fields': 'state,node,transport,noteto,url,support,mail-hosts,from,partner-address,to,proxy-url,hostname-subj,nht,perf,retry-count,reminder,max-http-size,max-smtp-size,remove-private-data,ondemand-server-url,support,reminder,ondemand-state,local-collection,validate-digital-certificate'}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error fetching info: %s' % error)
        for param in ('transport', 'mail_hosts', 'proxy_url', 'retry_count', 'max_http_size', 'max_smtp_size', 'noteto', 'validate_digital_certificate'):
            if param in record:
                asup_info[param] = record[param]
        asup_info['support'] = record['support'] in ['enable', True]
        asup_info['node_name'] = record['node'] if 'node' in record else ''
        asup_info['post_url'] = record['url'] if 'url' in record else ''
        asup_info['from_address'] = record['from'] if 'from' in record else ''
        asup_info['to_addresses'] = record['to'] if 'to' in record else list()
        asup_info['hostname_in_subject'] = record['hostname_subj'] if 'hostname_subj' in record else False
        asup_info['nht_data_enabled'] = record['nht'] if 'nht' in record else False
        asup_info['perf_data_enabled'] = record['perf'] if 'perf' in record else False
        asup_info['reminder_enabled'] = record['reminder'] if 'reminder' in record else False
        asup_info['private_data_removed'] = record['remove_private_data'] if 'remove_private_data' in record else False
        asup_info['local_collection_enabled'] = record['local_collection'] if 'local_collection' in record else False
        asup_info['ondemand_enabled'] = record['ondemand_state'] in ['enable', True] if 'ondemand_state' in record else False
        asup_info['service_state'] = 'started' if record['state'] in ['enable', True] else 'stopped'
        asup_info['partner_addresses'] = record['partner_address'] if 'partner_address' in record else list()
    else:
        asup_details = netapp_utils.zapi.NaElement('autosupport-config-get')
        asup_details.add_new_child('node-name', self.parameters['node_name'])
        try:
            result = self.server.invoke_successfully(asup_details, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching info: %s' % to_native(error), exception=traceback.format_exc())
        asup_attr_info = result.get_child_by_name('attributes').get_child_by_name('autosupport-config-info')
        asup_info['service_state'] = 'started' if asup_attr_info['is-enabled'] == 'true' else 'stopped'
        for item_key, zapi_key in self.na_helper.zapi_string_keys.items():
            value = asup_attr_info.get_child_content(zapi_key)
            asup_info[item_key] = value if value is not None else ''
        for item_key, zapi_key in self.na_helper.zapi_int_keys.items():
            value = asup_attr_info.get_child_content(zapi_key)
            if value is not None:
                asup_info[item_key] = self.na_helper.get_value_for_int(from_zapi=True, value=value)
        for item_key, zapi_key in self.na_helper.zapi_bool_keys.items():
            value = asup_attr_info.get_child_content(zapi_key)
            if value is not None:
                asup_info[item_key] = self.na_helper.get_value_for_bool(from_zapi=True, value=value)
        for item_key, zapi_key in self.na_helper.zapi_list_keys.items():
            parent, dummy = zapi_key
            asup_info[item_key] = self.na_helper.get_value_for_list(from_zapi=True, zapi_parent=asup_attr_info.get_child_by_name(parent))
    return asup_info