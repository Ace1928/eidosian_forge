from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_foreign_server_acls(self, fs):
    query = 'SELECT srvacl::text FROM pg_catalog.pg_foreign_server\n                   WHERE srvname = ANY (%s) ORDER BY srvname'
    self.execute(query, (fs,))
    return [t['srvacl'] for t in self.cursor.fetchall()]