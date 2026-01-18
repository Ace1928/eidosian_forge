from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native, to_bytes
from hashlib import sha1
def create_user_config(self, cursor):
    query_string = 'INSERT INTO mysql_users (\n               username,\n               backend,\n               frontend'
    cols = 3
    query_data = [self.username, self.backend, self.frontend]
    for col, val in iteritems(self.config_data):
        if val is not None:
            cols += 1
            query_data.append(val)
            query_string += ',\n' + col
    query_string += ')\n' + 'VALUES (' + '%s ,' * cols
    query_string = query_string[:-2]
    query_string += ')'
    cursor.execute(query_string, query_data)
    return True