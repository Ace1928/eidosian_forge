from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def dict_from_record(self, record):
    if not record:
        return None
    return_value = {'interface_name': record['name'], 'interface_type': self.parameters['interface_type'], 'uuid': record['uuid'], 'admin_status': 'up' if record['enabled'] else 'down'}
    if self.na_helper.safe_get(record, ['location', 'home_node', 'name']):
        return_value['home_node'] = record['location']['home_node']['name']
    if self.na_helper.safe_get(record, ['location', 'home_port', 'name']):
        return_value['home_port'] = record['location']['home_port']['name']
    if self.na_helper.safe_get(record, ['svm', 'name']):
        return_value['vserver'] = record['svm']['name']
    if 'data_protocol' in record:
        return_value['data_protocol'] = record['data_protocol']
    if 'auto_revert' in record['location']:
        return_value['is_auto_revert'] = record['location']['auto_revert']
    if 'failover' in record['location']:
        return_value['failover_scope'] = record['location']['failover']
    if self.na_helper.safe_get(record, ['ip', 'address']):
        return_value['address'] = netapp_ipaddress.validate_and_compress_ip_address(record['ip']['address'], self.module)
        if self.na_helper.safe_get(record, ['ip', 'netmask']) is not None:
            return_value['netmask'] = record['ip']['netmask']
    if self.na_helper.safe_get(record, ['service_policy', 'name']):
        return_value['service_policy'] = record['service_policy']['name']
    if self.na_helper.safe_get(record, ['location', 'node', 'name']):
        return_value['current_node'] = record['location']['node']['name']
    if self.na_helper.safe_get(record, ['location', 'port', 'name']):
        return_value['current_port'] = record['location']['port']['name']
    if self.na_helper.safe_get(record, ['dns_zone']):
        return_value['dns_domain_name'] = record['dns_zone']
    if self.na_helper.safe_get(record, ['probe_port']) is not None:
        return_value['probe_port'] = record['probe_port']
    if 'ddns_enabled' in record:
        return_value['is_dns_update_enabled'] = record['ddns_enabled']
    if self.na_helper.safe_get(record, ['subnet', 'name']):
        return_value['subnet_name'] = record['subnet']['name']
    return return_value