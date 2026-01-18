from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cifs_server_rest(self, from_name=None):
    """
        get details of the cifs_server.
        """
    if not self.use_rest:
        return self.get_cifs_server()
    query = {'svm.name': self.parameters['vserver'], 'fields': 'svm.uuid,enabled,security.smb_encryption,security.kdc_encryption,security.smb_signing,security.restrict_anonymous,'}
    query['name'] = from_name or self.parameters['cifs_server_name']
    api = 'protocols/cifs/services'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8):
        security_option_9_8 = 'security.encrypt_dc_connection,security.lm_compatibility_level,'
        query['fields'] += security_option_9_8
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
        security_option_9_10 = 'security.use_ldaps,security.use_start_tls,security.try_ldap_channel_binding,security.session_security,security.ldap_referral_enabled,security.aes_netlogon_enabled,'
        query['fields'] += security_option_9_10
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
        service_option_9_10 = 'options.multichannel,'
        query['fields'] += service_option_9_10
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error on fetching cifs: %s' % error)
    if record:
        record['service_state'] = 'started' if record.pop('enabled') else 'stopped'
        return {'svm': {'uuid': self.na_helper.safe_get(record, ['svm', 'uuid'])}, 'cifs_server_name': self.na_helper.safe_get(record, ['name']), 'service_state': self.na_helper.safe_get(record, ['service_state']), 'smb_signing': self.na_helper.safe_get(record, ['security', 'smb_signing']), 'encrypt_dc_connection': self.na_helper.safe_get(record, ['security', 'encrypt_dc_connection']), 'kdc_encryption': self.na_helper.safe_get(record, ['security', 'kdc_encryption']), 'smb_encryption': self.na_helper.safe_get(record, ['security', 'smb_encryption']), 'aes_netlogon_enabled': self.na_helper.safe_get(record, ['security', 'aes_netlogon_enabled']), 'ldap_referral_enabled': self.na_helper.safe_get(record, ['security', 'ldap_referral_enabled']), 'session_security': self.na_helper.safe_get(record, ['security', 'session_security']), 'lm_compatibility_level': self.na_helper.safe_get(record, ['security', 'lm_compatibility_level']), 'try_ldap_channel_binding': self.na_helper.safe_get(record, ['security', 'try_ldap_channel_binding']), 'use_ldaps': self.na_helper.safe_get(record, ['security', 'use_ldaps']), 'use_start_tls': self.na_helper.safe_get(record, ['security', 'use_start_tls']), 'restrict_anonymous': self.na_helper.safe_get(record, ['security', 'restrict_anonymous']), 'is_multichannel_enabled': self.na_helper.safe_get(record, ['options', 'multichannel'])}
    return record