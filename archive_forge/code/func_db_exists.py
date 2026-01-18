from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def db_exists(conn, cursor, db):
    cursor.execute('SELECT name FROM master.sys.databases WHERE name = %s', db)
    conn.commit()
    return bool(cursor.rowcount)