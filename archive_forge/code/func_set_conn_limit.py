from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def set_conn_limit(cursor, db, conn_limit):
    query = 'ALTER DATABASE "%s" CONNECTION LIMIT %s' % (db, conn_limit)
    executed_commands.append(query)
    cursor.execute(query)
    return True