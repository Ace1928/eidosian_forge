from __future__ import absolute_import, division, print_function
import os
import warnings
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def changeprimary(cursor, chm, connection_name='', channel=''):
    if connection_name:
        query = "CHANGE MASTER '%s' TO %s" % (connection_name, ','.join(chm))
    else:
        query = 'CHANGE MASTER TO %s' % ','.join(chm)
    if channel:
        query += " FOR CHANNEL '%s'" % channel
    executed_queries.append(query)
    cursor.execute(query)