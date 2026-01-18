from __future__ import absolute_import, division, print_function
import os
import warnings
from re import match
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.database import SQLParseError, mysql_quote_identifier
from ansible_collections.community.mysql.plugins.module_utils.mysql import mysql_connect, mysql_driver, mysql_driver_fail_msg, mysql_common_argument_spec
from ansible.module_utils._text import to_native
def check_mysqld_auto(module, cursor, mysqlvar):
    """Check variable's value in mysqld-auto.cnf."""
    query = 'SELECT VARIABLE_VALUE FROM performance_schema.persisted_variables WHERE VARIABLE_NAME = %s'
    try:
        cursor.execute(query, (mysqlvar,))
        res = cursor.fetchone()
    except Exception as e:
        if "Table 'performance_schema.persisted_variables' doesn't exist" in str(e):
            module.fail_json(msg='Server version must be 8.0 or greater.')
    if res:
        return res[0]
    else:
        return None