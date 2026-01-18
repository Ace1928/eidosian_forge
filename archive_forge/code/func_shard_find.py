from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def shard_find(client, shard):
    """Check if a shard exists.

    Args:
        client (cursor): Mongodb cursor on admin database.
        shard (str): shard to check.

    Returns:
        dict: when user exists, False otherwise.
    """
    if '/' in shard:
        s = shard.split('/')[0]
    else:
        s = shard
    for shard in client['config'].shards.find({'_id': s}):
        return shard
    return False