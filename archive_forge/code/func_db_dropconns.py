from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def db_dropconns(cursor, db):
    if get_server_version(cursor.connection) >= 90200:
        ' Drop DB connections in Postgres 9.2 and above '
        query_terminate = 'SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname=%(db)s AND pid <> pg_backend_pid()'
    else:
        ' Drop DB connections in Postgres 9.1 and below '
        query_terminate = 'SELECT pg_terminate_backend(pg_stat_activity.procpid) FROM pg_stat_activity WHERE pg_stat_activity.datname=%(db)s AND procpid <> pg_backend_pid()'
    query_block = 'UPDATE pg_database SET datallowconn = false WHERE datname=%(db)s'
    query = query_block + '; ' + query_terminate
    cursor.execute(query, {'db': db})