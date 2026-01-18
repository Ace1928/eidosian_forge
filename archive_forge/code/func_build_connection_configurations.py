from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def build_connection_configurations(self, provider_type, endpoints):
    """ Build "connection_configurations" objects from
        requested endpoints provided by user

        Returns:
            the user requested provider endpoints list
        """
    connection_configurations = []
    endpoint_keys = endpoint_list_spec().keys()
    provider_defaults = supported_providers().get(provider_type, {})
    endpoint = endpoints.get('provider')
    default_auth_key = endpoint.get('auth_key')
    for endpoint_key in endpoint_keys:
        endpoint = endpoints.get(endpoint_key)
        if endpoint:
            role = endpoint.get('role') or provider_defaults.get(endpoint_key + '_role', 'default')
            if role == 'default':
                authtype = provider_defaults.get('authtype') or role
            else:
                authtype = role
            connection_configurations.append({'endpoint': {'role': role, 'hostname': endpoint.get('hostname'), 'port': endpoint.get('port'), 'verify_ssl': [0, 1][endpoint.get('validate_certs', True)], 'security_protocol': endpoint.get('security_protocol'), 'certificate_authority': endpoint.get('certificate_authority'), 'path': endpoint.get('path')}, 'authentication': {'authtype': authtype, 'userid': endpoint.get('userid'), 'password': endpoint.get('password'), 'auth_key': endpoint.get('auth_key') or default_auth_key}})
    return connection_configurations