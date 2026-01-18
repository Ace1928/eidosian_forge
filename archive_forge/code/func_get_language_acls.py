from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_language_acls(self, languages):
    query = 'SELECT lanacl::text FROM pg_catalog.pg_language\n                   WHERE lanname = ANY (%s) ORDER BY lanname'
    self.execute(query, (languages,))
    return [t['lanacl'] for t in self.cursor.fetchall()]