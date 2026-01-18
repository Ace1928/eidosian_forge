from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_snmp_config_rest(self):
    """Retrieve cluster wide SNMP configuration"""
    fields = 'enabled'
    if self.parameters.get('auth_traps_enabled') is not None:
        fields += ',auth_traps_enabled'
    if 'traps_enabled' in self.parameters and self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
        fields += ',traps_enabled'
    record, error = rest_generic.get_one_record(self.rest_api, 'support/snmp', None, fields)
    if error:
        self.module.fail_json(msg='Error fetching SNMP configuration: %s' % to_native(error), exception=traceback.format_exc())
    if record:
        return {'enabled': record.get('enabled'), 'auth_traps_enabled': record.get('auth_traps_enabled'), 'traps_enabled': record.get('traps_enabled')}
    return None