from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def db_import(conn, cursor, module, db, target):
    if os.path.isfile(target):
        with open(target, 'r') as backup:
            sqlQuery = 'USE [%s]\n' % db
            for line in backup:
                if line is None:
                    break
                elif line.startswith('GO'):
                    cursor.execute(sqlQuery)
                    sqlQuery = 'USE [%s]\n' % db
                else:
                    sqlQuery += line
            cursor.execute(sqlQuery)
            conn.commit()
        return (0, 'import successful', '')
    else:
        return (1, 'cannot find target file', 'cannot find target file')