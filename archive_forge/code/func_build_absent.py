from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def build_absent(self):
    if self._obj_type == 'default_privs':
        self.query = []
        for obj in ['TABLES', 'SEQUENCES', 'TYPES']:
            if self._as_who:
                self.query.append('ALTER DEFAULT PRIVILEGES FOR ROLE {0}{1} REVOKE ALL ON {2} FROM {3};'.format(self._as_who, self._schema, obj, self._for_whom))
            else:
                self.query.append('ALTER DEFAULT PRIVILEGES{0} REVOKE ALL ON {1} FROM {2};'.format(self._schema, obj, self._for_whom))
    else:
        self.query.append('REVOKE {0} FROM {1};'.format(self._set_what, self._for_whom))