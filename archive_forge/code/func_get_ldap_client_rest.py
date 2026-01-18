from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_ldap_client_rest(self):
    """
        Retrives ldap client config with rest API.
        """
    if not self.use_rest:
        return self.get_ldap_client()
    query = {'svm.name': self.parameters.get('vserver'), 'fields': 'svm.uuid,ad_domain,servers,preferred_ad_servers,bind_dn,schema,port,base_dn,base_scope,min_bind_level,session_security,use_start_tls,'}
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 0):
        query['fields'] += 'bind_as_cifs_server,query_timeout,referral_enabled,ldaps_enabled'
    record, error = rest_generic.get_one_record(self.rest_api, 'name-services/ldap', query)
    if error:
        self.module.fail_json(msg='Error on getting idap client info: %s' % error)
    if record:
        return {'svm': {'uuid': self.na_helper.safe_get(record, ['svm', 'uuid'])}, 'ad_domain': self.na_helper.safe_get(record, ['ad_domain']), 'preferred_ad_servers': self.na_helper.safe_get(record, ['preferred_ad_servers']), 'servers': self.na_helper.safe_get(record, ['servers']), 'schema': self.na_helper.safe_get(record, ['schema']), 'port': self.na_helper.safe_get(record, ['port']), 'ldaps_enabled': self.na_helper.safe_get(record, ['ldaps_enabled']), 'min_bind_level': self.na_helper.safe_get(record, ['min_bind_level']), 'bind_dn': self.na_helper.safe_get(record, ['bind_dn']), 'base_dn': self.na_helper.safe_get(record, ['base_dn']), 'base_scope': self.na_helper.safe_get(record, ['base_scope']), 'use_start_tls': self.na_helper.safe_get(record, ['use_start_tls']), 'session_security': self.na_helper.safe_get(record, ['session_security']), 'referral_enabled': self.na_helper.safe_get(record, ['referral_enabled']), 'bind_as_cifs_server': self.na_helper.safe_get(record, ['bind_as_cifs_server']), 'query_timeout': self.na_helper.safe_get(record, ['query_timeout'])}
    return None