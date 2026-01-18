from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def set_tblspace(self, tblspace):
    query = 'ALTER TABLE %s SET TABLESPACE "%s"' % (pg_quote_identifier(self.name, 'table'), tblspace)
    return exec_sql(self, query, return_bool=True)