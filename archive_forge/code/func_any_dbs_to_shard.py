from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def any_dbs_to_shard(client, sharded_databases):
    """
    Return a list of databases that need to have sharding enabled
    sharded_databases - Provided by module
    cluster_sharded_databases - List of sharded dbs from the mongos
    """
    dbs_to_shard = []
    cluster_sharded_databases = sharded_dbs(client)
    for db in sharded_databases:
        if db not in cluster_sharded_databases:
            dbs_to_shard.append(db)
    return dbs_to_shard