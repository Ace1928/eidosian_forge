from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_all_tables_in_schema(self, schema):
    if schema:
        if not self.schema_exists(schema):
            raise Error('Schema "%s" does not exist.' % schema)
        query = "SELECT relname\n                       FROM pg_catalog.pg_class c\n                       JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n                       WHERE nspname = %s AND relkind in ('r', 'v', 'm', 'p')"
        self.execute(query, (schema,))
    else:
        query = "SELECT relname FROM pg_catalog.pg_class WHERE relkind in ('r', 'v', 'm', 'p')"
        self.execute(query)
    return [t['relname'] for t in self.cursor.fetchall()]