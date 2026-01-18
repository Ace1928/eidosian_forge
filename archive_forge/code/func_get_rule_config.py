from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def get_rule_config(self, cursor, created_rule_id=None):
    query_string = 'SELECT *\n               FROM mysql_query_rules'
    if created_rule_id:
        query_data = [created_rule_id]
        query_string += '\nWHERE rule_id = %s'
        cursor.execute(query_string, query_data)
        rule = cursor.fetchone()
    else:
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += '\n WHERE ' + col + ' = %s'
                else:
                    query_string += '\n  AND ' + col + ' = %s'
        if cols > 0:
            cursor.execute(query_string, query_data)
        else:
            cursor.execute(query_string)
        rule = cursor.fetchall()
    return rule