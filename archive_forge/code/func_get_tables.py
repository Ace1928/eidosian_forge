from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def get_tables(cursor):
    result = dict()
    tables = list()
    cursor.execute('show tables')
    for table in cursor.fetchall():
        tables.append(table.get('tables'))
    result['tables'] = tables
    for table in result.get('tables'):
        cursor.execute('select * from {table}'.format(table=table))
        if 'global_variables' in table:
            result[table] = dict()
            for item in cursor.fetchall():
                result[table][item.get('variable_name')] = item.get('variable_value')
        else:
            result[table] = cursor.fetchall()
    return result