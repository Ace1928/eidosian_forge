from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import (
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import \
def clientscopes_to_add(existing, proposed):
    to_add = []
    existing_clientscope_ids = extract_field(existing, 'id')
    for clientscope in proposed:
        if not clientscope['id'] in existing_clientscope_ids:
            to_add.append(clientscope)
    return to_add