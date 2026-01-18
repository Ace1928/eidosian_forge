from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def add_zone_tag(client, shard, tag):
    """
    Adds a tag to a shard
    @client - MongoDB connection
    @shard - The shard name
    @tag - The tag or Zone name
    """
    cmd_doc = OrderedDict([('addShardToZone', shard), ('zone', tag)])
    client['admin'].command(cmd_doc)