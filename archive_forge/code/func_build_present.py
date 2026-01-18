from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def build_present(self):
    if self._obj_type == 'default_privs':
        self.add_default_revoke()
        self.add_default_priv()
    else:
        self.query.append('GRANT {0} TO {1}'.format(self._set_what, self._for_whom))
        self.add_grant_option()