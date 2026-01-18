from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def sharded_dbs(client):
    """
    Returns the sharded databases
    Args:
        client (cursor): Mongodb cursor on admin database.
    Returns:
        a list of database names that are sharded
    """
    sharded_databases = []
    for entry in client['config'].databases.find({'partitioned': True}, {'_id': 1}):
        sharded_databases.append(entry['_id'])
    return sharded_databases