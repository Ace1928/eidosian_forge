from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def set_stor_params(self, params):
    query = 'ALTER TABLE %s SET (%s)' % (pg_quote_identifier(self.name, 'table'), params)
    return exec_sql(self, query, return_bool=True)