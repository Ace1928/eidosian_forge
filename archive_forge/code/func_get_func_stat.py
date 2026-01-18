from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_func_stat(self):
    """Get function statistics and fill out self.info dictionary."""
    query = 'SELECT * FROM pg_stat_user_functions'
    qp = None
    if self.schema:
        query = 'SELECT * FROM pg_stat_user_functions WHERE schemaname = %s'
        qp = (self.schema,)
    result = exec_sql(self, query, query_params=qp, add_to_executed=False)
    if not result:
        return
    self.__fill_out_info(result, info_key='functions', schema_key='schemaname', name_key='funcname')