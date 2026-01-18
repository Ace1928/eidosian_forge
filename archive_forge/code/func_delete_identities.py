from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def delete_identities(self, user, identities):
    changed = False
    for identity in user.identities:
        if identity not in identities:
            if not self._module.check_mode:
                user.identityproviders.delete(identity['provider'])
            changed = True
    return changed