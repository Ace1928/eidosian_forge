from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_security_config_rest(self):
    """
            Get the current security configuration
        """
    fields = 'fips.enabled,'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
        fields += 'tls.cipher_suites,tls.protocol_versions'
    record, error = rest_generic.get_one_record(self.rest_api, '/security', None, fields)
    if error:
        self.module.fail_json(msg='Error on getting security config: %s' % error)
    if record:
        return {'is_fips_enabled': self.na_helper.safe_get(record, ['fips', 'enabled']), 'supported_cipher_suites': self.na_helper.safe_get(record, ['tls', 'cipher_suites']), 'supported_protocols': self.na_helper.safe_get(record, ['tls', 'protocol_versions'])}
    return record