from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_interface_rest(self, name):
    """
        Return details about the interface
        :param:
            name : Name of the interface

        :return: Details about the interface. None if not found.
        :rtype: dict
        """
    self.derive_interface_type()
    if_type = self.parameters.get('interface_type')
    if 'vserver' in self.parameters:
        query_ip = {'name': name, 'svm.name': self.parameters['vserver']}
        query_fc = query_ip
    else:
        query_ip = {'name': '*%s' % name, 'scope': 'cluster'}
        query_fc = None
    fields = 'name,location,uuid,enabled,svm.name'
    fields_fc = fields + ',data_protocol'
    fields_ip = fields + ',ip,service_policy'
    if self.parameters.get('dns_domain_name'):
        fields_ip += ',dns_zone'
    if self.parameters.get('probe_port') is not None:
        fields_ip += ',probe_port'
    if self.parameters.get('is_dns_update_enabled') is not None:
        fields_ip += ',ddns_enabled'
    if self.parameters.get('subnet_name') is not None:
        fields_ip += ',subnet'
    records, error, records2, error2 = (None, None, None, None)
    if if_type in [None, 'ip']:
        records, error = self.get_interface_records_rest('ip', query_ip, fields_ip)
    if if_type in [None, 'fc'] and query_fc:
        records2, error2 = self.get_interface_records_rest('fc', query_fc, fields_fc)
    if records and records2:
        msg = 'Error fetching interface %s - found duplicate entries, please indicate interface_type.' % name
        msg += ' - ip interfaces: %s' % records
        msg += ' - fc interfaces: %s' % records2
        self.module.fail_json(msg=msg)
    if error is None and error2 is not None and records:
        error2 = None
    if error2 is None and error is not None and records2:
        error = None
    if error or error2:
        errors = [to_native(err) for err in (error, error2) if err]
        self.module.fail_json(msg='Error fetching interface details for %s: %s' % (name, ' - '.join(errors)), exception=traceback.format_exc())
    if records:
        self.set_interface_type('ip')
    if records2:
        self.set_interface_type('fc')
        records = records2
    record = self.find_exact_match(records, name) if records else None
    return self.dict_from_record(record)