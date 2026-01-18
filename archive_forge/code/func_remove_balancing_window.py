from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def remove_balancing_window(client):
    s = False
    result = client['config'].settings.update_one({'_id': 'balancer'}, {'$unset': {'activeWindow': True}})
    if result.modified_count == 1:
        s = True
    return s