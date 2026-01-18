from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def delete_server_config(self, cursor):
    query_string = 'DELETE FROM mysql_servers\n               WHERE hostgroup_id = %s\n                 AND hostname = %s\n                 AND port = %s'
    query_data = [self.hostgroup_id, self.hostname, self.port]
    cursor.execute(query_string, query_data)
    return True