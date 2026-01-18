from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_default_privs(self, schema, *args):
    if schema:
        query = 'SELECT defaclacl::text\n                       FROM pg_default_acl a\n                       JOIN pg_namespace b ON a.defaclnamespace=b.oid\n                       WHERE b.nspname = %s;'
        self.execute(query, (schema,))
    else:
        self.execute('SELECT defaclacl::text FROM pg_default_acl;')
    return [t['defaclacl'] for t in self.cursor.fetchall()]