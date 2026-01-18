from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def get_repl_group_config(self, cursor):
    query_string = 'SELECT *\n               FROM mysql_replication_hostgroups\n               WHERE writer_hostgroup = %s'
    query_data = [self.writer_hostgroup]
    cursor.execute(query_string, query_data)
    repl_group = cursor.fetchone()
    return repl_group