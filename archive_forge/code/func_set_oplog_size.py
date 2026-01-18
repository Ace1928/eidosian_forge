from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def set_oplog_size(client, oplog_size_mb):
    cmd_doc = OrderedDict([('replSetResizeOplog', 1), ('size', oplog_size_mb)])
    client['admin'].command(cmd_doc)