from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def disable_autosplit(client):
    client['config'].settings.update_one({'_id': 'autosplit'}, {'$set': {'enabled': False}}, upsert=True)