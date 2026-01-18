from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def get_schedule_config(self, cursor):
    query_string = 'SELECT *\n               FROM scheduler\n               WHERE active = %s\n                 AND interval_ms = %s\n                 AND filename = %s'
    query_data = [self.active, self.interval_ms, self.filename]
    for col, val in iteritems(self.config_data):
        if val is not None:
            query_data.append(val)
            query_string += '\n  AND ' + col + ' = %s'
    cursor.execute(query_string, query_data)
    schedule = cursor.fetchall()
    return schedule