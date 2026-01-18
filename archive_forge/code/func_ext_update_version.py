from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def ext_update_version(check_mode, cursor, ext, version):
    """Update extension version.

    Return True if success.

    Args:
      cursor (cursor) -- cursor object of psycopg library
      ext (str) -- extension name
      version (str) -- extension version
    """
    query = 'ALTER EXTENSION "%s" UPDATE' % ext
    params = {}
    if version != 'latest':
        query += ' TO %(ver)s'
        params['ver'] = version
    if not check_mode:
        cursor.execute(query, params)
    executed_queries.append(cursor.mogrify(query, params))
    return True