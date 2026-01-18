from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_ems_destination(self, name):
    api = 'support/ems/destinations'
    query = {'name': name, 'fields': 'type,destination,filters.name,certificate.ca,'}
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 1):
        query['fields'] += 'certificate.name,'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 12, 1):
        syslog_option_9_12 = 'syslog.transport,syslog.port,syslog.format.message,syslog.format.timestamp_override,syslog.format.hostname_override,'
        query['fields'] += syslog_option_9_12
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    self.fail_on_error(error, 'fetching EMS destination for %s' % name)
    if record:
        current = {'name': self.na_helper.safe_get(record, ['name']), 'type': self.na_helper.safe_get(record, ['type']), 'destination': self.na_helper.safe_get(record, ['destination']), 'filters': None, 'certificate': self.na_helper.safe_get(record, ['certificate', 'name']), 'ca': self.na_helper.safe_get(record, ['certificate', 'ca'])}
        if record.get('syslog') is not None:
            current['syslog'] = {'port': self.na_helper.safe_get(record, ['syslog', 'port']), 'transport': self.na_helper.safe_get(record, ['syslog', 'transport']), 'timestamp_format_override': self.na_helper.safe_get(record, ['syslog', 'format', 'timestamp_override']), 'hostname_format_override': self.na_helper.safe_get(record, ['syslog', 'format', 'hostname_override']), 'message_format': self.na_helper.safe_get(record, ['syslog', 'format', 'message'])}
        if current['type'] and '-' in current['type']:
            current['type'] = current['type'].replace('-', '_')
        if self.na_helper.safe_get(record, ['filters']):
            current['filters'] = [filter['name'] for filter in record['filters']]
        return current
    return None