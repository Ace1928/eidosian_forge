from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def create_repl_group(self, result, cursor):
    if not self.check_mode:
        result['changed'] = self.create_repl_group_config(cursor)
        result['msg'] = 'Added server to mysql_hosts'
        result['repl_group'] = self.get_repl_group_config(cursor)
        self.manage_config(cursor, result['changed'])
    else:
        result['changed'] = True
        result['msg'] = 'Repl group would have been added to' + ' mysql_replication_hostgroups, however' + ' check_mode is enabled.'