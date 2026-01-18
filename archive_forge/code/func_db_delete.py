from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def db_delete(conn, cursor, db):
    try:
        cursor.execute('ALTER DATABASE [%s] SET single_user WITH ROLLBACK IMMEDIATE' % db)
    except Exception:
        pass
    cursor.execute('DROP DATABASE [%s]' % db)
    return not db_exists(conn, cursor, db)