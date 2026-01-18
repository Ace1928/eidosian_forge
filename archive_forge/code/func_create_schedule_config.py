from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def create_schedule_config(self, cursor):
    query_string = 'INSERT INTO scheduler (\n               active,\n               interval_ms,\n               filename'
    cols = 0
    query_data = [self.active, self.interval_ms, self.filename]
    for col, val in iteritems(self.config_data):
        if val is not None:
            cols += 1
            query_data.append(val)
            query_string += ',\n' + col
    query_string += ')\n' + 'VALUES (%s, %s, %s' + ', %s' * cols + ')'
    cursor.execute(query_string, query_data)
    return True