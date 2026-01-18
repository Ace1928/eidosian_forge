from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import (
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import \
def clientscopes_to_delete(existing, proposed):
    to_delete = []
    proposed_clientscope_ids = extract_field(proposed, 'id')
    for clientscope in existing:
        if not clientscope['id'] in proposed_clientscope_ids:
            to_delete.append(clientscope)
    return to_delete