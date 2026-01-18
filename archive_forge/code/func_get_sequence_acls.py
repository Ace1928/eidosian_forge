from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_sequence_acls(self, schema, sequences):
    if schema:
        query = "SELECT relacl::text\n                       FROM pg_catalog.pg_class c\n                       JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n                       WHERE nspname = %s AND relkind = 'S' AND relname = ANY (%s)\n                       ORDER BY relname"
        self.execute(query, (schema, sequences))
    else:
        query = "SELECT relacl::text FROM pg_catalog.pg_class WHERE  relkind = 'S' AND relname = ANY (%s) ORDER BY relname"
        self.execute(query)
    return [t['relacl'] for t in self.cursor.fetchall()]