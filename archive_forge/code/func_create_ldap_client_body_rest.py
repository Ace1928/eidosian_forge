from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_ldap_client_body_rest(self, modify=None):
    """
        ldap client config body for create and modify with rest API.
        """
    config_options = ['ad_domain', 'servers', 'preferred_ad_servers', 'bind_dn', 'schema', 'port', 'base_dn', 'referral_enabled', 'ldaps_enabled', 'base_scope', 'bind_as_cifs_server', 'bind_password', 'min_bind_level', 'query_timeout', 'session_security', 'use_start_tls']
    processing_options = ['skip_config_validation']
    body = {}
    for key in config_options:
        if not modify and key in self.parameters:
            body[key] = self.parameters[key]
        elif modify and key in modify:
            body[key] = modify[key]
    for key in processing_options:
        if body and key in self.parameters:
            body[key] = self.parameters[key]
    return body