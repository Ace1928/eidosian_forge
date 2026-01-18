from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import (
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import \
def extract_field(dictionary, field='name'):
    return [cs[field] for cs in dictionary]