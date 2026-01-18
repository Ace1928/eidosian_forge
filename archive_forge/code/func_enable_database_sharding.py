from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def enable_database_sharding(client, database):
    """
    Enables sharding on a database
    Args:
        client (cursor): Mongodb cursor on admin database.
    Returns:
        true on success, false on failure
    """
    s = False
    db = client['admin'].command('enableSharding', database)
    if db:
        s = True
    return s