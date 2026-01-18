from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def set_owner(cursor, schema, owner):
    query = 'ALTER SCHEMA %s OWNER TO "%s"' % (pg_quote_identifier(schema, 'schema'), owner)
    cursor.execute(query)
    executed_queries.append(query)
    return True