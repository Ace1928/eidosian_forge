from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_rslot_info(self):
    """Get information about replication slots if exist."""
    res = self.__exec_sql("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'pg_replication_slots')")
    if not res[0]['exists']:
        return True
    query = 'SELECT slot_name, plugin, slot_type, database, active FROM pg_replication_slots'
    res = self.__exec_sql(query)
    if not res:
        return True
    rslot_dict = {}
    for i in res:
        rslot_dict[i['slot_name']] = dict(plugin=i['plugin'], slot_type=i['slot_type'], database=i['database'], active=i['active'])
    self.pg_info['repl_slots'] = rslot_dict