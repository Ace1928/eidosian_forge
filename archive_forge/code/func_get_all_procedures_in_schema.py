from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_all_procedures_in_schema(self, schema):
    if self.pg_version < 110000:
        raise Error('PostgreSQL version must be >= 11 for type=procedure. Exit')
    if schema:
        if not self.schema_exists(schema):
            raise Error('Schema "%s" does not exist.' % schema)
        query = "SELECT p.proname, oidvectortypes(p.proargtypes) ptypes FROM pg_catalog.pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE nspname = %s and p.prokind = 'p'"
        self.execute(query, (schema,))
    else:
        query = "SELECT p.proname, oidvectortypes(p.proargtypes) ptypes FROM pg_catalog.pg_proc p WHERE p.prokind = 'p'"
        self.execute(query)
    return ['%s(%s)' % (t['proname'], t['ptypes']) for t in self.cursor.fetchall()]