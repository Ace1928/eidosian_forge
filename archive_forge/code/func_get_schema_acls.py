from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_schema_acls(self, schemas):
    query = 'SELECT nspacl::text FROM pg_catalog.pg_namespace\n                   WHERE nspname = ANY (%s) ORDER BY nspname'
    self.execute(query, (schemas,))
    return [t['nspacl'] for t in self.cursor.fetchall()]