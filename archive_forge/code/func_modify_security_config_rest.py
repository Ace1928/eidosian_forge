from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_security_config_rest(self, modify):
    """
            Modify the current security configuration
        """
    body = {}
    if 'is_fips_enabled' in modify:
        body['fips.enabled'] = modify['is_fips_enabled']
    if 'supported_cipher_suites' in modify:
        body['tls.cipher_suites'] = modify['supported_cipher_suites']
    if 'supported_protocols' in modify:
        body['tls.protocol_versions'] = modify['supported_protocols']
    record, error = rest_generic.patch_async(self.rest_api, '/security', None, body)
    if error:
        self.module.fail_json(msg='Error on modifying security config: %s' % error)