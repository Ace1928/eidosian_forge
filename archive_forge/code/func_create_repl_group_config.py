from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def create_repl_group_config(self, cursor):
    query_string = 'INSERT INTO mysql_replication_hostgroups (\n               writer_hostgroup,\n               reader_hostgroup,\n               comment)\n               VALUES (%s, %s, %s)'
    query_data = [self.writer_hostgroup, self.reader_hostgroup, self.comment or '']
    cursor.execute(query_string, query_data)
    if self.check_type_support:
        self.update_check_type(cursor)
    return True