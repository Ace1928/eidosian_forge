from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_type_acls(self, schema, types):
    if schema:
        query = 'SELECT t.typacl::text FROM pg_catalog.pg_type t\n                       JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace\n                       WHERE n.nspname = %s AND t.typname = ANY (%s) ORDER BY typname'
        self.execute(query, (schema, types))
    else:
        query = 'SELECT typacl::text FROM pg_catalog.pg_type WHERE typname = ANY (%s) ORDER BY typname'
        self.execute(query)
    return [t['typacl'] for t in self.cursor.fetchall()]