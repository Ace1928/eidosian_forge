from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def get_olplog_size(client):
    return int(client['local'].command('collStats', 'oplog.rs')['maxSize']) / 1024 / 1024