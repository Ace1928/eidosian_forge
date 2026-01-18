from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import KeycloakAPI, camel, \
from ansible.module_utils.basic import AnsibleModule
from copy import deepcopy
def get_identity_provider_with_mappers(kc, alias, realm):
    idp = kc.get_identity_provider(alias, realm)
    if idp is not None:
        idp['mappers'] = sorted(kc.get_identity_provider_mappers(alias, realm), key=lambda x: x.get('name'))
    if idp is None:
        idp = {}
    return idp